"""A interactive visualizer program for high-quality DCT analysis
"""

# isort: skip_file
# autopep8: off
from __future__ import annotations

import argparse
import copy
import dataclasses
import enum
import json
import math
import pathlib
import typing

# NOTE: Make sure to import PyTorch before importing PyViewer-extended
import torch

import nfdpy
import numpy as np
import pydantic
import pyviewer_extended
import radpsd
import radpsd.signal as _signal
import radpsd.torch_util as _torch_util
import torchvision.io as io
import torchvision.transforms.v2.functional as F
import typing_extensions
from imgui_bundle import imgui, implot, hello_imgui

import lib.util as util
import lib.windowing as windowing
# autopep8: on

logger = util.get_logger()

# --------------------------------------------------------------------------------------------------------------------------------------------------------
# Load the C++/CUDA module

# Check PyTorch is compiled with CUDA and nvcc is available
cuda_enabled = torch.cuda.is_available() and _torch_util.get_extension_loader()._check_command('nvcc')

# Build and load the C++/CUDA module to compute the radial power spectral density
logger.info('Loading the C++/CUDA module...')
_module = _signal._get_cpp_module(is_cuda=cuda_enabled, with_omp=True)
logger.info('done.')

# --------------------------------------------------------------------------------------------------------------------------------------------------------
# Device specific settings

if cuda_enabled:
    _device = torch.device('cuda')
    _dtype = torch.float64

    _n_radial_divs = 360 * 2
    _n_polar_divs = 1024

    DEFAULT_IMG_SIZE = 256
else:
    _device = torch.device('cpu')
    _dtype = torch.float64

    _n_radial_divs = 64
    _n_polar_divs = 64

    DEFAULT_IMG_SIZE = 128

logger.info(f'Using device: {_device}, dtype: {_dtype}')


# --------------------------------------------------------------------------------------------------------------------------------------------------------
# Parameters


class InterpMethod(enum.IntEnum):
    # autopep8: off
    nearest                  = 0
    bilinear                 = 1
    # autopep8: on

    def to_torch(self) -> F.InterpolationMode:
        return [
            F.InterpolationMode.NEAREST,
            F.InterpolationMode.BILINEAR,
        ][self.value]


class ImageMode(enum.IntEnum):
    # autopep8: off
    sinusoidal             = 0
    file                   = 1
    file_RGB               = 2
    gaussian               = 3
    gaussian_RGB           = 4
    # autopep8: on

    def __str__(self) -> str:
        return self.value


MPL_CMAPS: typing.Final[list[str]] = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'gray']

# Window functions
WINDOW_FUNC_NAMES = list(windowing._window_func_names)
WINDOW_FUNC_NAMES.remove('rectangular')  # Remove rectangular window

# Example images
EXAMPLE_IMG_DIR: typing.Final[pathlib.Path] = pathlib.Path(__file__).parent / 'example_imgs'

EXAMPLE_GRAY_IMG_PATH: typing.Final[pathlib.Path] = EXAMPLE_IMG_DIR / 'checkerboard_1024.png'
DEFAULT_GRAY_IMG_PATH: typing.Final[pathlib.Path] = EXAMPLE_GRAY_IMG_PATH if EXAMPLE_GRAY_IMG_PATH.is_file() else pathlib.Path()

EXAMPLE_RGB_IMG_PATH: typing.Final[pathlib.Path] = EXAMPLE_IMG_DIR / 'Hannusmetsä.png'
DEFAULT_RGB_IMG_PATH: typing.Final[pathlib.Path] = EXAMPLE_RGB_IMG_PATH if EXAMPLE_RGB_IMG_PATH.is_file() else pathlib.Path()

# Parameter class


@pydantic.dataclasses.dataclass(config=pydantic.config.ConfigDict(arbitrary_types_allowed=True))
class Params:
    """Parameters for the Color of Noise visualizer.
    """

    # autopep8: off
    img_mode                            : int          = int(ImageMode.sinusoidal)                                              # image mode

    wave_number                         : float        = 20.0                                                                   # wave number

    img_path                            : pathlib.Path = DEFAULT_RGB_IMG_PATH                                                   # image path

    seed                                : int          = 0                                                                      # random seed for the image generation
    sigma                               : float        = 1.0                                                                    # standard deviation for the Gaussian noise

    img_size                            : int          = DEFAULT_IMG_SIZE                                                       # image size
    rotate                              : float        = 0.0                                                                    # rotation angle in degrees

    enable_super_sampling               : bool         = False                                                                  # enable super sampling
    super_sampling_factor               : int          = 4                                                                      # super sampling factor

    enable_pre_filering                 : bool         = False                                                                  # enable pre-filtering
    pre_filter_padding                  : int          = 8                                                                      # padding for pre-filtering

    kernel_size                         : int          = 15                                                                     # kernel size for pre-filtering
    kernel_sigma                        : float        = 0.3 * 6 + 0.8                                                          # sigma for pre-filtering. See also: https://docs.pytorch.org/vision/main/generated/torchvision.transforms.functional.gaussian_blur.html

    enable_windowing                    : bool         = False                                                                  # switch for windowing
    window_func                         : int          = WINDOW_FUNC_NAMES.index('Kaiser')                                      # windowing function

    apply_padding                       : bool         = False                                                                  # apply zero padding to the image
    padding_factor                      : int          = 4                                                                      # padding factor for the FFT

    img_cmap_id                         : int          = 5                                                                      # color map ID for the sinusoidal image, by default 'gray'
    psd_cmap_id                         : int          = 1                                                                      # color map ID for the power spectral density, by default 'plasma'

    psd_profile_ylim_fixed              : bool         = False                                                                  # fix the y-limits of the radial and axial power spectral density plot
    psd_profile_ylim_min                : float        = 1e-12                                                                  # minimum y-limit for the radial and axial power spectral density plot
    psd_profile_ylim_max                : float        = 1e2                                                                    # maximum y-limit for the radial and axial power spectral density plot
    psd_profile_xscale_log              : bool         = False                                                                  # show x axis in log10 scale?
    psd_profile_yscale_log              : bool         = True                                                                   # show y axis in log10 scale?

    windowfn_instances                  : dict[str, windowing.WindowFunctionBase] = pydantic.Field(default_factory=dict)        # window functions, instantiated in the __post_init__ method
    # autopep8: on

    def __post_init__(self):
        # Instantiate all the window functions
        for name, cls in windowing._window_funcs.items():
            self.windowfn_instances[name] = cls()

    IO_BOUND = [int, float, bool, pathlib.Path]

    def load(self, params: dict) -> Params:
        for field in dataclasses.fields(Params):
            param_name = field.name
            param_value = getattr(self, param_name)

            if param_name not in params or param_value is None:
                continue

            for target_type in self.IO_BOUND:
                if isinstance(param_value, target_type):
                    if isinstance(param_value, pathlib.Path):
                        param_value = pathlib.Path(params[param_name])
                    else:
                        param_value = params[param_name]

                    setattr(self, param_name, param_value)
                    break

        return self

    def dump(self) -> dict:
        params_dict = {}

        for field in dataclasses.fields(Params):
            param_name = field.name
            param_value = getattr(self, param_name)

            if param_value is None:
                continue

            for target_type in self.IO_BOUND:
                if isinstance(param_value, target_type):
                    if isinstance(param_value, pathlib.Path):
                        # Convert pathlib.Path to str for json compatibility
                        param_value = str(param_value)

                    params_dict[param_name] = param_value
                    break

        return params_dict


@pydantic.dataclasses.dataclass(config=pydantic.dataclasses.ConfigDict(arbitrary_types_allowed=True))
class ImageFile:
    img: torch.Tensor
    file_path: pathlib.Path
    img_read_mode: io.ImageReadMode


# --------------------------------------------------------------------------------------------------------------------------------------------------------
# Viewer


class FFTVisualizer(pyviewer_extended.MultiTexturesDockingViewer):
    """Visualizer class for FFT
    """

    # ------------------------------------------------------------------------------------
    # Constants

    # autopep8: off
    KEY_INPUT         : typing.Final[str] = 'Input'
    KEY_MASKED        : typing.Final[str] = 'Masked'
    KEY_MASKED_INPUT  : typing.Final[str] = 'Masked Input'

    KEYS                                  = [KEY_INPUT, KEY_MASKED, KEY_MASKED_INPUT]

    MIN_IMG_SIZE      : int               = 5    # it has to larger than 4
    MAX_IMG_SIZE      : int               = 2048 # 256 is already heavy with padding even if using CUDA backend ...

    PARAMS_CACHE_PATH : pathlib.Path      = pathlib.Path('.cache/fft_viz_params.json')

    NUM_PARAMS_CACHES : int               = 9 # shuould be <10 due to the limit of number keys
    # autopep8: on

    # ------------------------------------------------------------------------------------

    def __init__(self, name: str, enable_vsync: bool, cache_params: bool = False):
        self.cur_cache_param_id = 0
        self.cached_params: list[Params] = [Params() for _ in range(self.NUM_PARAMS_CACHES)]

        self.cache_params_to_file = cache_params

        self.base_img: ImageFile | None = None

        # ------------------------------------------------------------------------------------
        super().__init__(
            name=name,
            texture_names=self.KEYS,
            enable_vsync=enable_vsync,
            full_screen_mode=hello_imgui.FullScreenMode.full_monitor_work_area,
            with_font_awesome=True,
            with_implot=True
        )

    @typing_extensions.override
    def setup_state(self) -> None:
        """Initialize the state of the visualizer. Called by the super class.
        """

        self.state.params = Params()
        self.state.prev_params = None

        self.state.img = None
        self.state.window_img = None
        self.state.windowed_img = None
        self.state.psd_img = None

        self.state.window = None
        self.state.rad_psd = None
        self.state.axial_psd_h = None
        self.state.axial_psd_v = None

        self.state.params_cache_file = self.PARAMS_CACHE_PATH

        # Load params
        if self.cache_params_to_file and self.PARAMS_CACHE_PATH.is_file():
            self.load_param_caches(self.PARAMS_CACHE_PATH)

        # Update param cache
        self.cached_params[self.cur_cache_param_id] = self.state.params

    def load_param_caches(self, file_path: pathlib.Path) -> None:
        with open(file_path, mode='r') as file:
            params = json.load(file)

        if params is not None and isinstance(params, list):
            for i_param, param_dict in enumerate(params):
                if i_param >= len(self.cached_params):
                    continue
                self.cached_params[i_param].load(param_dict)

            self.state.params = self.cached_params[self.cur_cache_param_id]

            print(f'Loaded parameters from {file_path}')
        else:
            print(f'Failed to load parameters from {file_path}')

    def save_param_caches(self, file_path: pathlib.Path) -> None:
        params: list[dict] = []
        for cached_param in self.cached_params:
            params.append(cached_param.dump())

        cache_file = file_path.resolve()
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        with open(cache_file, mode='w', encoding='utf8') as file:
            json.dump(params, file, indent=4, ensure_ascii=False)

        print(f'Saved parameters to {cache_file}')

    @typing_extensions.override
    def save_settings(self) -> None:
        if self.cache_params_to_file:
            self.save_param_caches(self.PARAMS_CACHE_PATH)

    @typing_extensions.override
    def drag_and_drop_callback(self, paths):
        if paths is not None and len(paths) > 0:
            cache_path: pathlib.Path = paths[0]
            if cache_path.is_file():
                self.load_param_caches(cache_path)

    @property
    def params(self) -> Params | None:
        """UI state parameters
        """

        state = getattr(self, 'state', None)

        return state.params if state is not None else None

    @property
    def img_cmap(self) -> str:
        if self.params is not None:
            return MPL_CMAPS[self.params.img_cmap_id]

        return 'viridis'

    @property
    def psd_cmap(self) -> str:
        if self.params is not None:
            return MPL_CMAPS[self.params.psd_cmap_id]

        return 'viridis'

    @typing_extensions.override
    def compute(self) -> dict[str, np.ndarray]:
        # Check if the parameters have changed
        if self.state.prev_params is not None and self.state.prev_params == self.params:
            return {
                self.KEY_INPUT: self.state.img,
                self.KEY_MASKED: self.state.window_img,
                self.KEY_MASKED_INPUT: self.state.windowed_img,
            }

        # ---------------------------------------------------------------------------------------------------
        # Compute a sinusoidal image

        wave_number = self.params.wave_number
        target_freq = wave_number / self.params.img_size  # [cycles/pixel]

        if (self.params.img_mode == ImageMode.file or self.params.img_mode == ImageMode.file_RGB) and self.params.img_path.is_file():
            # From file

            super_sampling_factor = 1  # disable super sampling for file images

            img_read_mode = io.ImageReadMode.RGB if self.params.img_mode == ImageMode.file_RGB else io.ImageReadMode.GRAY

            # Buffer the image data to avoid reading it from file multiple times
            if self.base_img is None or self.base_img.file_path != self.params.img_path or self.base_img.img_read_mode != img_read_mode:
                img_decoded = io.decode_image(self.params.img_path, mode=img_read_mode)
                img_decoded = F.to_dtype(img_decoded, scale=True).to(dtype=_dtype, device=_device)
                self.base_img = ImageFile(img=img_decoded, file_path=self.params.img_path, img_read_mode=img_read_mode)

            img = self.base_img.img  # [channels, img_size, img_size]

            # Rotate the image
            img = F.rotate_image(
                img,
                angle=-self.params.rotate,
                center=None,  # Center the image
                interpolation=F.InterpolationMode.BILINEAR,
                expand=False,  # Do not expand the image
            )
        elif (self.params.img_mode == ImageMode.gaussian or self.params.img_mode == ImageMode.gaussian_RGB):
            super_sampling_factor = 1  # disable super sampling for file images

            num_channels = 3 if self.params.img_mode == ImageMode.gaussian_RGB else 1

            generator = torch.Generator(device=_device).manual_seed(self.params.seed)
            img = self.params.sigma * torch.randn((num_channels, self.params.img_size, self.params.img_size), dtype=_dtype, device=_device, generator=generator)  # zero DC
        else:
            super_sampling_factor = self.params.super_sampling_factor if self.params.enable_super_sampling else 1

            # Allocate larger canvas for the rotation
            canvas_size = self.params.img_size

            n_pixels = canvas_size * super_sampling_factor
            w = 1.0 / (super_sampling_factor * 2.0)
            m = 0.5 - w

            t = torch.linspace(0.0 - m, canvas_size - 1.0 + m, n_pixels, dtype=_dtype, device=_device)
            t = t - t.mean()  # Center the grid around zero
            grid = torch.stack(torch.meshgrid(t, t, indexing='ij'), dim=-1)  # [n_pixels, n_pixels, 2]

            rad_angle = math.pi * self.params.rotate / 180.0
            rot_mat = torch.tensor([
                [math.cos(rad_angle), -math.sin(rad_angle)],
                [math.sin(rad_angle), math.cos(rad_angle)]
            ], dtype=_dtype, device=_device)  # [2, 2]

            grid = torch.einsum('ij,...j->...i', rot_mat, grid)  # [n_pixels, n_pixels, 2]

            x = torch.sin(2.0 * np.pi * target_freq * grid)
            img = x[..., 1]  # [n_pixels, n_pixels]

            # Add batch dimension
            img = img.unsqueeze(0)

        # Check image shape
        assert img.ndim == 3
        assert img.shape[0] in [1, 3]

        if self.params.enable_pre_filering:
            # Prefilering
            # Pad the image to avoid edge artifacts
            pad_size = self.params.pre_filter_padding
            img = F.pad_image(img, padding=(pad_size, pad_size, pad_size, pad_size), padding_mode='reflect')

            # Apply Gaussian blur
            img = F.gaussian_blur(
                img,
                kernel_size=(self.params.kernel_size, self.params.kernel_size),
                sigma=(self.params.kernel_sigma, self.params.kernel_sigma)
            )

            # Remove the padding
            if pad_size > 0:
                img = img[:, pad_size:-pad_size, pad_size:-pad_size]

        # Center crop the image
        img = F.center_crop(img, (self.params.img_size * super_sampling_factor, self.params.img_size * super_sampling_factor))
        img = F.resize(img, size=(self.params.img_size, self.params.img_size), interpolation=F.InterpolationMode.NEAREST, antialias=False)

        # Images for visualization
        img_plot = img.detach().clone().to(dtype=torch.float32, device=_device)  # [c, h, w]
        img_plot = util.normalize_0_to_1(img_plot, based_on_min_max=True)

        if img.shape[0] == 3:
            img_plot = img_plot.permute(1, 2, 0)
        elif self.img_cmap == 'gray':
            img_plot = img_plot.tile(3, 1, 1).permute(1, 2, 0)
        else:
            img_plot = radpsd.apply_color_map(img_plot, self.img_cmap).squeeze(0)

        # Sanity check
        assert img_plot.ndim == 3
        assert img_plot.shape[-1] == 3
        self.state.img = img_plot

        # ---------------------------------------------------------------------------------------------------
        # Compute FFT

        # Apply windowing
        window_name = WINDOW_FUNC_NAMES[self.params.window_func] if self.params.enable_windowing else 'rectangular'  # rectangular window is no-op
        window_func = self.params.windowfn_instances[window_name]
        window, window_2d = window_func.calc_window(img.shape[-1], dtype=img.dtype, device=img.device)

        window_2d = window_2d.unsqueeze(0)  # add channel dimension

        img = img * window_2d

        # For visualization
        window_2d_img = window_2d.detach().clone().tile(3, 1, 1).permute(1, 2, 0).to(torch.float32)  # [win_h, win_w, 3]
        window_2d_img = util.normalize_0_to_1(window_2d_img, based_on_min_max=True)

        self.state.window = np.ascontiguousarray(window.cpu().numpy().astype(np.float64))
        self.state.window_img = window_2d_img

        if self.params.apply_padding:
            # Apply zero padding
            padding = (self.params.img_size * self.params.padding_factor - self.params.img_size)
            img = torch.nn.functional.pad(img, (0, padding, 0, padding))

        # For visualization
        windowed_img = img.detach().clone().to(dtype=torch.float32, device=_device)  # [c, h, w]
        windowed_img = util.normalize_0_to_1(windowed_img, based_on_min_max=True)

        if img.shape[0] == 3:
            windowed_img = windowed_img.permute(1, 2, 0)
        elif self.img_cmap == 'gray':
            windowed_img = windowed_img.tile(3, 1, 1).permute(1, 2, 0)
        else:
            windowed_img = radpsd.apply_color_map(windowed_img, self.img_cmap).squeeze(0)

        # Sanity check
        assert windowed_img.ndim == 3
        assert windowed_img.shape[-1] == 3
        self.state.windowed_img = windowed_img

        # Compute power spectral density
        spectrum: torch.Tensor = torch.fft.fftn(img, dim=(-2, -1)).abs().square()
        spectrum = torch.fft.fftshift(spectrum, dim=(-2, -1))
        psd = spectrum / (img.shape[-2] * img.shape[-1])

        # Axial psd
        half_h = img.shape[-2] // 2
        half_w = img.shape[-1] // 2
        self.state.axial_psd_h = np.ascontiguousarray(np.fft.ifftshift(psd[:, half_h, :].cpu().numpy(), axes=1)[:, :half_w], dtype=np.float64)  # [C, W]
        self.state.axial_psd_v = np.ascontiguousarray(np.fft.ifftshift(psd[:, :, half_w].cpu().numpy(), axes=1)[:, :half_h], dtype=np.float64)  # [C, H]

        # 2D psd plot
        psd_plot = psd.mean(dim=0)  # take average on channel axis
        psd_plot = 10.0 * torch.log10(psd_plot + 1e-10)  # to decibels

        # Sanity check
        assert psd_plot.ndim == 2
        self.state.psd_img = np.ascontiguousarray(psd_plot.cpu().numpy()).astype(np.float32)

        # ---------------------------------------------------------------------------------------------------
        # Compute the radial power spectral density
        rad_psd = _module.calc_radial_psd(
            psd.permute(1, 2, 0).unsqueeze(0).contiguous(),  # [1, H, W, C]
            _n_radial_divs,
            _n_polar_divs,
        )  # [1, n_divs, n_points, C]

        rad_psd = rad_psd.mean(dim=(0, 1))  # [n_points, C]

        self.state.rad_psd = rad_psd.cpu().numpy().astype(np.float64)

        # ---------------------------------------------------------------------------------------------------
        # Copy parameters
        self.state.prev_params = copy.deepcopy(self.params)

        return {
            self.KEY_INPUT: self.state.img,
            self.KEY_MASKED: self.state.window_img,
            self.KEY_MASKED_INPUT: self.state.windowed_img,
        }

    def open_img_file_dialog(self) -> None:
        """Open an image file dialog to select an image.
        """

        default_path = str(self.params.img_path) if self.params.img_path.is_file() else str(pathlib.Path.cwd())
        img_path = nfdpy.open_file_dialog(filters={'Image file': 'png,jpg'}, default_path=default_path)
        if img_path is not None:
            self.params.img_path = pathlib.Path(img_path)

    def open_param_cache_file_dialog(self) -> None:
        if self.state.params_cache_file.is_file():
            cache_file = self.state.params_cache_file.resolve()
            default_path = str(cache_file.parent)
            default_name = cache_file.name
        else:
            default_path = str(pathlib.Path.cwd())
            default_name = 'params.json'

        cache_path = nfdpy.save_file_dialog(filters={'Param file': 'json'}, default_path=default_path, default_name=default_name)
        if cache_path is not None:
            self.state.params_cache_file = pathlib.Path(cache_path)
            self.save_param_caches(self.state.params_cache_file)

    @pyviewer_extended.dockable
    def psd_plot(self) -> None:
        # ---------------------------------------------------------------------------------------------------
        # Plot the power spectral density

        x_avail, _ = imgui.get_content_region_avail()
        color_bar_prop = 0.10  # 10% for the color bar
        plot_width = x_avail * (1.0 - color_bar_prop)
        color_bar_width = x_avail - plot_width

        cmap = getattr(implot.Colormap_, self.psd_cmap, None)
        if cmap is not None:
            implot.push_colormap(cmap.value)

        if self.state.psd_img is not None:
            psd_img = self.state.psd_img
            scale_min = np.min(psd_img)
            scale_max = np.max(psd_img)

            if implot.begin_plot(
                'Power Spectral Density',
                size=(plot_width, -1),
                flags=implot.Flags_.no_legend.value | implot.Flags_.equal.value
            ):

                half_size = self.params.img_size // 2

                implot.setup_axes("Frequency [wave number]", "Frequency [wave number]")
                implot.setup_axes_limits(-half_size, half_size, -half_size, half_size)
                implot.plot_heatmap(
                    label_id='Heat Power Spectral Density',
                    values=psd_img,
                    scale_min=scale_min,
                    scale_max=scale_max,
                    bounds_min=implot.Point(-half_size, -half_size),
                    bounds_max=implot.Point(half_size, half_size),
                    label_fmt='',
                )

                implot.end_plot()

            imgui.same_line()
            implot.colormap_scale("Power Spectral Density [dB]", scale_min, scale_max, size=(color_bar_width, -1))

        if cmap is not None:
            implot.pop_colormap()

    @pyviewer_extended.dockable
    def psd_profile_plot(self):
        if imgui.begin_tab_bar('Power Spectral Density Profiles'):
            # ----------------------------------------------------------------------------------------------------------------------------------------------------
            # Plot the radial power spectral density

            if imgui.begin_tab_item_simple('Radial'):
                if self.state.rad_psd is not None and implot.begin_plot(
                    'Radial Power Spectral Density',
                    size=(-1, -1),
                    flags=implot.Flags_.no_legend.value if self.state.rad_psd.shape[1] == 1 else 0
                ):
                    # Setup x axis
                    implot.setup_axis(implot.ImAxis_.x1, "Frequency [wave number]")
                    implot.setup_axis_limits(implot.ImAxis_.x1, 1.0, self.params.img_size / 2, imgui.Cond_.always.value)

                    # Setup y axis
                    implot.setup_axis(implot.ImAxis_.y1, "Power Spectral Density [dB]", flags=implot.AxisFlags_.auto_fit.value)
                    if self.params.psd_profile_ylim_fixed:
                        implot.setup_axis_limits(implot.ImAxis_.y1, self.params.psd_profile_ylim_min, self.params.psd_profile_ylim_max, imgui.Cond_.always.value)

                    # Set log-log scale
                    implot.setup_axis_scale(implot.ImAxis_.x1, implot.Scale_.log10.value if self.params.psd_profile_xscale_log else implot.Scale_.linear.value)
                    implot.setup_axis_scale(implot.ImAxis_.y1, implot.Scale_.log10.value if self.params.psd_profile_yscale_log else implot.Scale_.linear.value)

                    freq = radpsd.radial_freq(img_size=self.params.img_size, n_radial_bins=len(self.state.rad_psd), dtype=np.float64)  # [cycles/pix]
                    freq = freq * self.params.img_size  # ranged in [0, ..., img_size / 2]
                    freq = np.ascontiguousarray(freq[1:])  # w/o DC

                    if self.state.rad_psd.shape[1] == 3:
                        # RGB
                        implot.set_next_line_style(imgui.ImVec4(1.0, 0.0, 0.0, 1.0))
                        implot.plot_line('Radial PSD - Red', freq, np.ascontiguousarray(self.state.rad_psd[1:, 0]))
                        implot.set_next_line_style(imgui.ImVec4(0.0, 1.0, 0.0, 1.0))
                        implot.plot_line('Radial PSD - Green', freq, np.ascontiguousarray(self.state.rad_psd[1:, 1]))
                        implot.set_next_line_style(imgui.ImVec4(0.0, 0.0, 1.0, 1.0))
                        implot.plot_line('Radial PSD - Blue', freq, np.ascontiguousarray(self.state.rad_psd[1:, 2]))
                    else:
                        implot.plot_line('Radial Power Spectral Density', freq, np.ascontiguousarray(self.state.rad_psd[1:, 0]))

                    implot.end_plot()
                imgui.end_tab_item()

            # ----------------------------------------------------------------------------------------------------------------------------------------------------
            # Plot the axial power spectral density

            for label, axial_psd in (('Horizontal', self.state.axial_psd_h), ('Vertical', self.state.axial_psd_v)):
                if imgui.begin_tab_item_simple(label):
                    # Plot the axial power spectral density
                    if axial_psd is not None and self.state.psd_img is not None and implot.begin_plot(
                        f'{label} Power Spectral Density',
                        size=(-1, -1),
                        flags=implot.Flags_.no_legend.value if axial_psd.shape[0] == 1 else 0
                    ):
                        padded_img_size = self.state.psd_img.shape[-1]
                        freq = np.fft.fftfreq(padded_img_size, d=1.0/self.params.img_size)[1:axial_psd.shape[1]].astype(np.float64)  # assume that the img is square

                        # Setup x axis
                        implot.setup_axis(implot.ImAxis_.x1, "Frequency [wave number]")
                        implot.setup_axis_limits(implot.ImAxis_.x1, freq.min(), freq.max(), imgui.Cond_.always.value)

                        # Setup y axis
                        implot.setup_axis(implot.ImAxis_.y1, "Power Spectral Density [dB]", flags=implot.AxisFlags_.auto_fit.value)
                        if self.params.psd_profile_ylim_fixed:
                            implot.setup_axis_limits(implot.ImAxis_.y1, self.params.psd_profile_ylim_min, self.params.psd_profile_ylim_max, imgui.Cond_.always.value)

                        # Set log-log scale
                        implot.setup_axis_scale(implot.ImAxis_.x1, implot.Scale_.log10.value if self.params.psd_profile_xscale_log else implot.Scale_.linear.value)
                        implot.setup_axis_scale(implot.ImAxis_.y1, implot.Scale_.log10.value if self.params.psd_profile_yscale_log else implot.Scale_.linear.value)

                        if axial_psd.shape[0] == 3:
                            # RGB
                            implot.set_next_line_style(imgui.ImVec4(1.0, 0.0, 0.0, 1.0))
                            implot.plot_line(f'{label} PSD - Red', freq, np.ascontiguousarray(axial_psd[0, 1:]))
                            implot.set_next_line_style(imgui.ImVec4(0.0, 1.0, 0.0, 1.0))
                            implot.plot_line(f'{label} PSD - Green', freq, np.ascontiguousarray(axial_psd[1, 1:]))
                            implot.set_next_line_style(imgui.ImVec4(0.0, 0.0, 1.0, 1.0))
                            implot.plot_line(f'{label} PSD - Blue', freq, np.ascontiguousarray(axial_psd[2, 1:]))
                        else:
                            implot.plot_line('Horizontal Power Spectral Density', freq, np.ascontiguousarray(axial_psd[0, 1:]))

                        implot.end_plot()
                    imgui.end_tab_item()
            imgui.end_tab_bar()

    @pyviewer_extended.dockable
    def toolbar(self) -> None:
        """Build the toolbar UI for the Color of Noise visualizer.
        """

        # ---------------------------------------------------------------------------------------------------
        # Parameter caching

        # Detect arrow key
        is_any_item_active = imgui.is_any_item_active()
        if not is_any_item_active:  # Except when inputting values ...
            if self.keyhit(imgui.Key.right_arrow):
                self.cur_cache_param_id = (self.cur_cache_param_id + 1) % self.NUM_PARAMS_CACHES
            elif not imgui.is_any_item_active() and self.keyhit(imgui.Key.left_arrow):
                self.cur_cache_param_id = (self.cur_cache_param_id - 1) % self.NUM_PARAMS_CACHES

        # Detect number key
        for i_cache in range(self.NUM_PARAMS_CACHES):
            key = getattr(imgui.Key, f'_{i_cache + 1}')
            if not is_any_item_active and self.keyhit(key):  # Except when inputting values ...
                self.cur_cache_param_id = i_cache

        imgui.separator_text('Current Param')
        for i_cache in range(self.NUM_PARAMS_CACHES):
            if i_cache != 0:
                imgui.same_line()

            if imgui.radio_button(f'##Param{i_cache}', self.cur_cache_param_id == i_cache) or self.cur_cache_param_id == i_cache:
                self.cur_cache_param_id = i_cache
                self.state.params = self.cached_params[self.cur_cache_param_id]
        imgui.text('Tips: switch them with "arrow" or "number" keys')

        # ---------------------------------------------------------------------------------------------------
        # Parameters for the noise

        imgui.separator()
        if imgui.collapsing_header('Input', flags=imgui.TreeNodeFlags_.default_open):
            imgui.separator_text('Image')

            self.params.img_mode = imgui.combo(
                'Image mode',
                self.params.img_mode,
                [m.name for m in ImageMode],
            )[1]

            if self.params.img_mode == ImageMode.sinusoidal:
                min_wave_number = 0.0
                max_wave_number = float(self.params.img_size)

                imgui.text('Wave number Slider')
                imgui.same_line()
                self.params.wave_number = imgui.slider_float('##Wave number slider', self.params.wave_number, min_wave_number, max_wave_number)[1]

                imgui.text('Wave number Input ')
                imgui.same_line()
                self.params.wave_number = imgui.input_float('##Wave number input', self.params.wave_number)[1]

                # Validate
                self.params.wave_number = max(min(self.params.wave_number, max_wave_number), min_wave_number)
            elif self.params.img_mode == ImageMode.gaussian or self.params.img_mode == ImageMode.gaussian_RGB:
                self.params.seed = imgui.slider_int('Random seed', self.params.seed, 0, 10000)[1]
                self.params.sigma = imgui.slider_float('Standard deviation', self.params.sigma, 0.0, 100.0)[1]
            else:
                self.params.img_path = pathlib.Path(imgui.input_text('Image path', str(self.params.img_path))[1])
                imgui.same_line()
                imgui.push_id('Open img file selection dialog')
                if imgui.button('Open'):
                    self.open_img_file_dialog()
                imgui.pop_id()

            imgui.separator_text('Image size')
            self.params.img_size = imgui.input_int('##Image size input', self.params.img_size)[1]
            self.params.img_size = imgui.slider_int('##Image size slider', self.params.img_size, self.MIN_IMG_SIZE, self.MAX_IMG_SIZE)[1]
            self.params.img_size = max(min(self.params.img_size, self.MAX_IMG_SIZE), self.MIN_IMG_SIZE)

            imgui.separator_text('Geometric transformation')
            self.params.rotate = imgui.slider_float('Rotation angle', self.params.rotate, 0.0, 360.0)[1]

            if imgui.collapsing_header('Super sampling'):
                if self.params.img_mode == ImageMode.sinusoidal:
                    self.params.enable_super_sampling = imgui.checkbox('Enable super sampling', self.params.enable_super_sampling)[1]
                    self.params.super_sampling_factor = imgui.slider_int('Super sampling factor', self.params.super_sampling_factor, 1, 4)[1]
                else:
                    imgui.text('Super sampling is disabled for file images')

            if imgui.collapsing_header('Pre-filtering'):
                self.params.enable_pre_filering = imgui.checkbox('Enable pre-filtering', self.params.enable_pre_filering)[1]
                self.params.pre_filter_padding = imgui.slider_int('Padding (reflection)', self.params.pre_filter_padding, 0, 32)[1]
                kernel_size = imgui.slider_int('Kernel size', self.params.kernel_size, 3, 31)[1]
                self.params.kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
                self.params.kernel_sigma = imgui.slider_float('Kernel sigma', self.params.kernel_sigma, 0.01, 10.0)[1]

        # ---------------------------------------------------------------------------------------------------
        # FFT parameters

        imgui.separator()
        if imgui.collapsing_header('Windowing', flags=imgui.TreeNodeFlags_.default_open):
            imgui.push_id('enable_windowing')
            self.params.enable_windowing = imgui.checkbox('Enable', self.params.enable_windowing)[1]
            imgui.pop_id()

            if self.params.enable_windowing:
                self.params.window_func = imgui.combo(
                    'Window function',
                    self.params.window_func,
                    WINDOW_FUNC_NAMES,
                )[1]

                imgui.separator_text('Window parameters')
                window_name = WINDOW_FUNC_NAMES[self.params.window_func]
                window_func = self.params.windowfn_instances[window_name]
                for name, param in window_func.params.items():
                    param.draw_param_widgets(name)

                imgui.separator()
                if self.state.window is not None and implot.begin_plot(f'{window_name} Window', size=(-1, 256)):
                    implot.setup_axes('Pixel', 'Weight')
                    implot.setup_axes_limits(
                        0.0,
                        self.state.window.shape[0] - 1,
                        -0.3,
                        1.3,
                        imgui.Cond_.always.value,
                    )
                    implot.plot_line(f'{window_name} window', self.state.window)
                    implot.end_plot()

        imgui.separator()
        if imgui.collapsing_header('Padding', flags=imgui.TreeNodeFlags_.default_open):
            imgui.push_id('enable_padding')
            self.params.apply_padding = imgui.checkbox('Enable', self.params.apply_padding)[1]
            imgui.pop_id()

            if self.params.apply_padding:
                self.params.padding_factor = imgui.slider_int('Padding factor', self.params.padding_factor, 1, 16)[1]

        # ---------------------------------------------------------------------------------------------------
        # Visualization parameters

        imgui.separator()
        if imgui.collapsing_header('Visualization', flags=imgui.TreeNodeFlags_.default_open):
            imgui.separator_text('Color maps')
            self.params.img_cmap_id = imgui.combo('Image color map', self.params.img_cmap_id, MPL_CMAPS)[1]
            self.params.psd_cmap_id = imgui.combo('PSD color map', self.params.psd_cmap_id, MPL_CMAPS)[1]

            imgui.separator_text('Power Spectral Density Profile')
            self.params.psd_profile_ylim_fixed = imgui.checkbox('Fix y-limits', self.params.psd_profile_ylim_fixed)[1]
            if self.params.psd_profile_ylim_fixed:
                self.params.psd_profile_ylim_min = imgui.input_float('Min y-limit', self.params.psd_profile_ylim_min, step=1e-10, format='%.2e')[1]
                self.params.psd_profile_ylim_max = imgui.input_float('Max y-limit', self.params.psd_profile_ylim_max, step=1e-10, format='%.2e')[1]
            self.params.psd_profile_xscale_log = imgui.checkbox('X in log scale', self.params.psd_profile_xscale_log)[1]
            self.params.psd_profile_yscale_log = imgui.checkbox('Y in log scale', self.params.psd_profile_yscale_log)[1]

        imgui.separator()

        # ---------------------------------------------------------------------------------------------------
        # Reset button

        imgui.separator()
        if imgui.button('Reset params', size=(-1, 40)):
            self.state.params = Params()

        # ---------------------------------------------------------------------------------------------------
        if self.cache_params_to_file:
            imgui.separator()
            self.state.params_cache_file = pathlib.Path(imgui.input_text('Param cache', str(self.state.params_cache_file))[1])
            imgui.same_line()
            imgui.push_id('Open param cache file selection dialog')
            if imgui.button('Open'):
                self.open_param_cache_file_dialog()
            imgui.pop_id()
            imgui.same_line()
            if imgui.button('Save'):
                self.save_param_caches(self.state.params_cache_file)

        # ---------------------------------------------------------------------------------------------------
        # Cache param
        self.cached_params[self.cur_cache_param_id] = self.params

    @typing_extensions.override
    def setup_docking_layout(self, layout_funcs):
        # Graceful initial layout ----------------------------------------------------------------------------------------------------------------
        # Window layout
        #   + ---------------------------- + ---------------------------- + ---------------------------- + ---------------------------- +
        #   | 'psd_plot'                   | 'Input'                                                     |                              |
        #   + ---------------------------- + ---------------------------- + ---------------------------- + 'toolbar'                    |
        #   | 'psd_profile_plot'           | 'Masked'                     | 'Masked Input'               |                              |
        #   + ---------------------------- + ---------------------------- + ---------------------------- + ---------------------------- |
        #
        # Dockspace names
        #   + ---------------------------- + ---------------------------- + ---------------------------- + ---------------------------- +
        #   | 'MainDockSpace'              | 'Dock0'                                                     |                              |
        #   + ---------------------------- + ---------------------------- + ---------------------------- + 'Dock1'                      |
        #   | 'Dock2'                      | 'Dock3'                      | 'Dock4'                      |                              |
        #   + ---------------------------- + ---------------------------- + ---------------------------- + ---------------------------- |

        splits = [
            # autopep8: off
            # Split colomns first
            hello_imgui.DockingSplit('MainDockSpace', 'Dock0', imgui.Dir.right, ratio_=3/4),
            hello_imgui.DockingSplit(        'Dock0', 'Dock1', imgui.Dir.right, ratio_=1/3),

            hello_imgui.DockingSplit('MainDockSpace', 'Dock2',  imgui.Dir.down, ratio_=1/2),

            hello_imgui.DockingSplit(        'Dock0', 'Dock3',  imgui.Dir.down, ratio_=1/2),
            hello_imgui.DockingSplit(        'Dock3', 'Dock4', imgui.Dir.right, ratio_=1/2),
            # autopep8: on
        ]

        title_to_dockspace_name = {
            # autopep8: off
                           'psd_plot' : 'MainDockSpace',          'Input' : 'Dock0',      'toolbar' : 'Dock1',
                   'psd_profile_plot' :         'Dock2',         'Masked' : 'Dock3', 'Masked Input' : 'Dock4',
            # autopep8: on
        }

        windows = [hello_imgui.DockableWindow(f._title, title_to_dockspace_name[f._title], f, can_be_closed_=True) for f in layout_funcs]

        return splits, windows


# -------------------------------------------------------------------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='A high-quality FFT visualization tool',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--no_cache',
        action='store_true',
        help='Disable caching parameters (except imgui parameters)'
    )
    parser.add_argument(
        '--vsync',
        action='store_true',
        help='Enable VSync'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    vis = FFTVisualizer('FFT Vis', enable_vsync=args.vsync, cache_params=not args.no_cache)
    logger.info('Bye!')

# -------------------------------------------------------------------------------------------------------------------------------------------------
