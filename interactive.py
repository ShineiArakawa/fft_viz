# isort: skip_file
# autopep8: off
import copy
import enum
import math
import pathlib
import typing

# NOTE: Make sure to import PyTorch before importing PyViewer-extended
import torch

import cv2
import nfdpy
import numpy as np
import pydantic
import pyviewer_extended
import radpsd
import radpsd.torch_util as _torch_util
import radpsd.signal as _signal
import torchvision.transforms.v2.functional as F
from imgui_bundle import imgui, implot

import util
import windowing
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

    def to_cv2(self) -> int:
        return [
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
        ][self.value]


class ImageMode(enum.IntEnum):
    # autopep8: off
    sinusoidal             = 0
    file                   = 1
    # autopep8: on

    def __str__(self) -> str:
        return self.value


MPL_CMAPS: typing.Final[list[str]] = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'gray']

EXAMPLE_IMG_PATH: typing.Final[pathlib.Path] = pathlib.Path(__file__).parent / 'example_imgs' / 'checkerboard_1024.png'
DEFAULT_IMG_PATH: typing.Final[pathlib.Path] = EXAMPLE_IMG_PATH if EXAMPLE_IMG_PATH.is_file() else pathlib.Path()


@pydantic.dataclasses.dataclass(config=pydantic.config.ConfigDict(arbitrary_types_allowed=True))
class Params:
    """Parameters for the Color of Noise visualizer.
    """

    # autopep8: off
    img_mode                            : int          = int(ImageMode.sinusoidal)                                              # image mode
    img_path                            : pathlib.Path = DEFAULT_IMG_PATH                                                       # image path (Default is a FFHQ image)
    img_size                            : int          = DEFAULT_IMG_SIZE                                                       # image size
    enable_super_sampling               : bool         = False                                                                  # enable super sampling
    super_sampling_factor               : int          = 4                                                                      # super sampling factor
    enable_pre_filering                 : bool         = False                                                                  # enable pre-filtering
    pre_filter_padding                  : int          = 8                                                                      # padding for pre-filtering
    kernel_size                         : int          = 15                                                                     # kernel size for pre-filtering
    kernel_sigma                        : float        = 0.3 * 6 + 0.8                                                          # sigma for pre-filtering. See also: https://docs.pytorch.org/vision/main/generated/torchvision.transforms.functional.gaussian_blur.html
    wave_number                         : float        = 20.0                                                                   # wave number
    rotate                              : float        = 0.0                                                                    # rotation angle in degrees

    window_func                         : int          = 0                                                                      # windowing function
    apply_padding                       : bool         = False                                                                  # apply zero padding to the image
    padding_factor                      : int          = 4                                                                      # padding factor for the FFT

    img_cmap_id                         : int          = 5                                                                      # color map ID for the sinusoidal image, by default 'gray'
    psd_cmap_id                         : int          = 1                                                                      # color map ID for the power spectrum density, by default 'plasma'

    windowfn_instances                  : dict[str, windowing.WindowFunctionBase] = pydantic.Field(default_factory=dict)        # window functions, instantiated in the __post_init__ method
    # autopep8: on

    def __post_init__(self):
        # Instantiate all the window functions
        for name, cls in windowing._window_funcs.items():
            self.windowfn_instances[name] = cls()


@pydantic.dataclasses.dataclass(config=pydantic.dataclasses.ConfigDict(arbitrary_types_allowed=True))
class ImageFile:
    img: torch.Tensor
    file_path: pathlib.Path


# --------------------------------------------------------------------------------------------------------------------------------------------------------
# utilities


def to_cuda_device(x: torch.Tensor) -> torch.Tensor:
    """Copy the input tensor to the CUDA device if available.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor to be copied.

    Returns
    -------
    torch.Tensor
        Copied tensor on the CUDA device if available, otherwise the original tensor.
    """

    if torch.cuda.is_available():
        return x.cuda()

    return x

# --------------------------------------------------------------------------------------------------------------------------------------------------------
# Viewer


class FFTVisualizer(pyviewer_extended.MultiTexturesDockingViewer):
    """Visualizer class for FFT
    """

    # autopep8: off
    KEY_INPUT        : typing.Final[str] = 'Input'
    KEY_MASKED       : typing.Final[str] = 'Masked'
    KEY_MASKED_INPUT : typing.Final[str] = 'Masked Input'
    # autopep8: on

    KEYS = [KEY_INPUT, KEY_MASKED, KEY_MASKED_INPUT]

    def __init__(self, name):
        self.base_img: ImageFile | None = None

        # ------------------------------------------------------------------------------------
        super().__init__(name, self.KEYS, with_font_awesome=True, with_implot=True)

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

        if self.params.img_mode == ImageMode.file and self.params.img_path.is_file():
            super_sampling_factor = 1  # disable super sampling for file images

            if self.base_img is None or self.base_img.file_path != self.params.img_path:
                img = cv2.imread(str(self.params.img_path), cv2.IMREAD_GRAYSCALE)
                self.base_img = ImageFile(img=torch.tensor(img).to(dtype=_dtype, device=_device), file_path=self.params.img_path)

            img = self.base_img.img

            # Add batch dimension
            img = img.unsqueeze(0)

            # Rotate the image
            img = F.rotate_image(
                img,
                angle=-self.params.rotate,
                center=None,  # Center the image
                interpolation=F.InterpolationMode.BILINEAR,
                expand=False,  # Do not expand the image
            )
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
        img = img.squeeze(0)

        # Images for visualization
        img_plot = img.detach().clone().to(torch.float32)
        img_plot = util.normalize_0_to_1(img_plot, based_on_min_max=True)
        if self.img_cmap == 'gray':
            img_plot = torch.stack([img_plot, img_plot, img_plot], dim=-1)
        else:
            img_plot = radpsd.apply_color_map(img_plot, self.img_cmap)

        self.state.img = img_plot

        # ---------------------------------------------------------------------------------------------------
        # Compute FFT

        # Apply windowing
        window_name = windowing._window_func_names[self.params.window_func]
        window_func = self.params.windowfn_instances[window_name]

        window = window_func.calc_window(img.shape[-1], dtype=img.dtype, device=img.device)
        window_2d = torch.ger(window, window)  # [short_side, short_side]

        img = img * window_2d

        # For visualization
        window_2d_img = window_2d.detach().clone().unsqueeze(-1).tile(1, 1, 3).to(torch.float32)
        window_2d_img = util.normalize_0_to_1(window_2d_img, based_on_min_max=True)

        self.state.window = np.ascontiguousarray(window.cpu().numpy().astype(np.float64))
        self.state.window_img = window_2d_img

        if self.params.apply_padding:
            # Apply zero padding

            padding = (self.params.img_size * self.params.padding_factor - self.params.img_size)

            img = torch.nn.functional.pad(img, (0, padding, 0, padding))

        # For visualization
        windowed_img = img.detach().clone().to(dtype=torch.float32, device=_device)
        windowed_img = util.normalize_0_to_1(windowed_img, based_on_min_max=True)
        if self.img_cmap == 'gray':
            windowed_img = torch.stack([windowed_img, windowed_img, windowed_img], dim=-1)
        else:
            windowed_img = radpsd.apply_color_map(windowed_img, self.img_cmap)
        self.state.windowed_img = windowed_img

        # Compute power spectrum density
        spectrum: torch.Tensor = torch.fft.fftn(img, dim=(-2, -1)).abs().square()
        spectrum = torch.fft.fftshift(spectrum, dim=(-2, -1))
        psd = spectrum / (img.shape[-2] * img.shape[-1])

        psd_db = 10.0 * torch.log10(psd + 1e-10)  # to decibels

        self.state.psd_img = np.ascontiguousarray(psd_db.detach().cpu().numpy()).astype(np.float32)

        # ---------------------------------------------------------------------------------------------------
        # Compute the radial power spectrum density
        rad_psd = _module.calc_radial_psd(
            psd.unsqueeze(0).unsqueeze(-1).contiguous(),  # [1, H, W, 1]
            _n_radial_divs,
            _n_polar_divs,
        )  # [1, n_divs, n_points, 1]

        rad_psd = rad_psd.mean(dim=(0, 1, 3))  # [n_points,]

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

    @pyviewer_extended.dockable
    def psd_plot(self) -> None:
        # ---------------------------------------------------------------------------------------------------
        # Plot the power spectrum density

        cmap = getattr(implot.Colormap_, self.psd_cmap, None)
        if cmap is not None:
            implot.push_colormap(cmap.value)

        heatmap_size = 128 * 5
        if self.state.psd_img is not None and implot.begin_plot(
            'Power Spectrum Density',
            size=(-1, -1),
            flags=implot.Flags_.no_legend.value | implot.Flags_.equal.value
        ):
            psd_img = self.state.psd_img

            scale_min = np.min(psd_img)
            scale_max = np.max(psd_img)

            half_size = self.params.img_size // 2

            implot.setup_axes("Frequency [wave number]", "Frequency [wave number]")
            implot.setup_axes_limits(-half_size, half_size, -half_size, half_size)
            implot.plot_heatmap(
                label_id='Head Power Spectrum Density',
                values=psd_img,
                scale_min=scale_min,
                scale_max=scale_max,
                bounds_min=implot.Point(-half_size, -half_size),
                bounds_max=implot.Point(half_size, half_size),
                label_fmt='',
            )

            imgui.same_line()
            implot.colormap_scale("Power Spectrum Density [dB]", scale_min, scale_max, size=(20, heatmap_size))

            implot.end_plot()

        if cmap is not None:
            implot.pop_colormap()

    @pyviewer_extended.dockable
    def radial_psd_plot(self):
        # ---------------------------------------------------------------------------------------------------
        # Plot the radial power spectrum density

        if self.state.rad_psd is not None and imgui.same_line() is None and implot.begin_plot(
            'Radial Power Spectrum Density',
            size=(-1, -1),
            flags=implot.Flags_.no_legend.value
        ):
            implot.setup_axis(implot.ImAxis_.x1, "Frequency [wave number]")
            implot.setup_axis_limits(implot.ImAxis_.x1, 0.0, self.params.img_size / 2, imgui.Cond_.always.value)

            implot.setup_axis(implot.ImAxis_.y1, "Power Spectrum Density [dB]", flags=implot.AxisFlags_.auto_fit.value)

            # log-log scale
            implot.setup_axis_scale(implot.ImAxis_.x1, implot.Scale_.linear)
            implot.setup_axis_scale(implot.ImAxis_.y1, implot.Scale_.log10)

            x_value = np.linspace(0.0, self.params.img_size / 2, len(self.state.rad_psd), dtype=np.float64)
            implot.plot_line('Radial Power Spectrum Density', np.ascontiguousarray(x_value[1:]), np.ascontiguousarray(self.state.rad_psd[1:]))

            implot.end_plot()

    @pyviewer_extended.dockable
    def toolbar(self) -> None:
        """Build the toolbar UI for the Color of Noise visualizer.
        """

        # ---------------------------------------------------------------------------------------------------
        # Parameters for the noise
        if imgui.collapsing_header('Input', flags=imgui.TreeNodeFlags_.default_open):
            imgui.separator_text('Image')

            self.params.img_mode = imgui.combo(
                'Image mode',
                self.params.img_mode,
                [m.name for m in ImageMode],
            )[1]

            if self.params.img_mode == ImageMode.sinusoidal:
                imgui.text('Wave number Slider')
                imgui.same_line()
                self.params.wave_number = imgui.slider_float('##Wave number slider', self.params.wave_number, 0.0, float(self.params.img_size))[1]

                imgui.text('Wave number Input ')
                imgui.same_line()
                self.params.wave_number = imgui.input_float('##Wave number input', self.params.wave_number)[1]
            else:
                self.params.img_path = pathlib.Path(imgui.input_text('Image path', str(self.params.img_path))[1])
                imgui.same_line()
                if imgui.button('Open'):
                    self.open_img_file_dialog()

            self.params.img_size = imgui.slider_int('Image size', self.params.img_size, 16, 1024)[1]

            imgui.separator_text('Geometric transformation')
            self.params.rotate = imgui.slider_float('Rotation angle', self.params.rotate, 0.0, 360.0)[1]

            imgui.separator_text('Super sampling')
            if self.params.img_mode != ImageMode.file:
                self.params.enable_super_sampling = imgui.checkbox('Enable super sampling', self.params.enable_super_sampling)[1]
                self.params.super_sampling_factor = imgui.slider_int('Super sampling factor', self.params.super_sampling_factor, 1, 4)[1]
            else:
                imgui.text('Super sampling is disabled for file images')

            imgui.separator_text('Pre-filtering')
            self.params.enable_pre_filering = imgui.checkbox('Enable pre-filtering', self.params.enable_pre_filering)[1]
            self.params.pre_filter_padding = imgui.slider_int('Padding (reflection)', self.params.pre_filter_padding, 0, 32)[1]
            kernel_size = imgui.slider_int('Kernel size', self.params.kernel_size, 3, 31)[1]
            self.params.kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
            self.params.kernel_sigma = imgui.slider_float('Kernel sigma', self.params.kernel_sigma, 0.01, 10.0)[1]

        # ---------------------------------------------------------------------------------------------------
        # FFT parameters

        if imgui.collapsing_header('Windowing', flags=imgui.TreeNodeFlags_.default_open):
            self.params.window_func = imgui.combo(
                'Window function',
                self.params.window_func,
                windowing._window_func_names,
            )[1]

            imgui.separator_text('Window parameters')
            window_name = windowing._window_func_names[self.params.window_func]
            window_func = self.params.windowfn_instances[window_name]
            for name, param in window_func.params.items():
                param.add_slider_and_input(name)

            imgui.separator()
            if self.state.window is not None and implot.begin_plot(f'{window_name} Window', size=(-1, 256)):
                implot.setup_axes('Pixel', 'Weight')
                implot.setup_axes_limits(
                    0.0,
                    self.state.window.shape[0] - 1,
                    - 0.3,
                    1.3,
                    imgui.Cond_.always.value,
                )
                implot.plot_line(f'{window_name} window', self.state.window)
                implot.end_plot()

        if imgui.collapsing_header('Padding', flags=imgui.TreeNodeFlags_.default_open):
            imgui.push_id('enable_padding')
            self.params.apply_padding = imgui.checkbox('Enable', self.params.apply_padding)[1]
            imgui.pop_id()
            self.params.padding_factor = imgui.slider_int('Padding factor', self.params.padding_factor, 1, 16)[1]

            imgui.separator_text('Memo')
            imgui.text(f'Niquist frequency: 0.5 [cycles/pixel], {self.params.img_size // 2} [wave number]')

        # ---------------------------------------------------------------------------------------------------
        # Visualization parameters
        if imgui.collapsing_header('Visualization', flags=imgui.TreeNodeFlags_.default_open):
            self.params.img_cmap_id = imgui.combo('Image color map', self.params.img_cmap_id, MPL_CMAPS)[1]
            self.params.psd_cmap_id = imgui.combo('PSD color map', self.params.psd_cmap_id, MPL_CMAPS)[1]

        imgui.separator()

        # ---------------------------------------------------------------------------------------------------
        if imgui.button('Reset all params', size=(-1, 40)):
            self.state.params = Params()


if __name__ == '__main__':
    _ = FFTVisualizer('FFT')
    logger.info('Bye!')
