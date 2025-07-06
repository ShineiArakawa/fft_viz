"""
fir_viz
=======

A interactive visualization program for high-quality finite impulse response filter design

Dev log
-------
5.7.2025 : Implemented minimum requirements

TO DO
-----
1. 入力画像のプロット x
2. 入力画像のPSD x
3. 入力画像のPSDプロファイル x
4. フィルターのプロット x
5. フィルターのプロファイル x
6. フィルターされたフーリエ係数のPSDプロット x
7. フィルターされたフーリエ係数のプロファイル x
8. 出力画像のプロット x

9. PSD関係の計算にはwindowingとpaddingの切り替えができるようにする (必要ないかも)

## Filter

1. 正規化方法の見直し
   normで割るのが妥当 x
2. フィルターの追加
   1. Butterworth
   2. Chebyshev
   3. Elliptic
   4. Kaiser (優先度は低め)
3. 2次元展開方法の提案
   1. 外積法
   2. Exactサンプリング
4. 高品質なフィルタ設計
   1. パディングによる高周波成分の抑制

## その他

1. Performance tuning

参考リンク
https://scikit-image.org/docs/0.25.x/auto_examples/filters/plot_butterworth.html

Contact
-------

- Author: Shinei Arakawa
- Email: arakawashinei1115@gmail.com
"""

# autopep8: off
# isort: skip_file

from __future__ import annotations

import argparse
import copy
import dataclasses
import enum
import json
import math
import pathlib
import typing
import uuid

# NOTE: Make sure to import PyTorch before importing PyViewer-extended
import torch

import cachetools
import nfdpy
import numpy as np
import pydantic
import pyviewer_extended
import pyviewer_extended.multi_textures_viewer as multi_textures_viewer
import radpsd
import radpsd.signal as _signal
import radpsd.torch_util as _torch_util
import torchvision.io as io
import torchvision.transforms.v2.functional as F
import typing_extensions
from imgui_bundle import imgui, implot

import lib.util as util
import lib.filtering as filtering
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

    img_cmap_id                         : int          = MPL_CMAPS.index('gray')                                                # color map ID for the sinusoidal image, by default 'gray'
    psd_cmap_id                         : int          = MPL_CMAPS.index('plasma')                                              # color map ID for the power spectral density, by default 'plasma'
    filter_cmap_id                      : int          = MPL_CMAPS.index('plasma')                                              # color map ID for the filter, by default 'plasma'

    input_psd_profile_ylim_fixed        : bool         = False                                                                  # fix the y-limits of the radial and axial power spectral density plot
    input_psd_profile_ylim_min          : float        = 1e-12                                                                  # minimum y-limit for the radial and axial power spectral density plot
    input_psd_profile_ylim_max          : float        = 1e2                                                                    # maximum y-limit for the radial and axial power spectral density plot
    input_psd_profile_xscale_log        : bool         = False                                                                  # show x axis in log10 scale?
    input_psd_profile_yscale_log        : bool         = True                                                                   # show y axis in log10 scale?

    filter_response_ylim_fixed          : bool         = True                                                                   # fix the y-limits of the radial and axial power spectral density plot
    filter_response_ylim_min            : float        = -10.0                                                                  # minimum y-limit for the radial and axial power spectral density plot
    filter_response_ylim_max            : float        = 0.5                                                                    # maximum y-limit for the radial and axial power spectral density plot
    filter_response_xscale_log          : bool         = False                                                                  # show x axis in log10 scale?

    filtered_psd_profile_ylim_fixed     : bool         = False                                                                  # fix the y-limits of the radial and axial power spectral density plot
    filtered_psd_profile_ylim_min       : float        = 1e-12                                                                  # minimum y-limit for the radial and axial power spectral density plot
    filtered_psd_profile_ylim_max       : float        = 1e2                                                                    # maximum y-limit for the radial and axial power spectral density plot
    filtered_psd_profile_xscale_log     : bool         = False                                                                  # show x axis in log10 scale?
    filtered_psd_profile_yscale_log     : bool         = True                                                                   # show y axis in log10 scale?

    filters                             : list         = pydantic.Field(default_factory=list)
    retain_power                        : bool         = False
    # autopep8: on

    IO_BOUND = [int, float, bool, pathlib.Path, list]

    def load(self, params: dict) -> Params:
        for param_key, param_value in params.items():
            # Check param_key is in fields
            target_field = None
            for field in dataclasses.fields(Params):
                if param_key == field.name and any([field.type == bound.__name__ for bound in self.IO_BOUND]):
                    target_field = field
                    break

            if target_field is None:
                # Unknown key or out of bound
                continue

            if field.type is pathlib.Path:
                param_value = pathlib.Path(param_value)
                setattr(self, field.name, param_value)
            elif field.type == 'list' and isinstance(param_value, list) and field.name == 'filters':
                for filter_param in param_value:
                    if filter_class := filter_param.pop('filter_class', None):
                        # Instantiate
                        filter_inst = None
                        for filter_t in filtering.filters.values():
                            if filter_class == filter_t.__name__:
                                filter_inst = filter_t()
                                break

                        if filter_inst is not None:
                            # Override params
                            for key, value in filter_param.items():
                                if key in filter_inst.params:
                                    filter_inst.params[key].value = filter_inst.params[key].value_type(value)
                            self.filters.append(filter_inst)
            else:
                setattr(self, field.name, param_value)

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
                        param_value_out = str(param_value)
                        params_dict[param_name] = param_value_out
                    elif isinstance(param_value, list):
                        param_value_out = []
                        for filter_inst in param_value:
                            if isinstance(filter_inst, filtering.FilterBase):
                                params: dict[str, util.ValueEntity] = filter_inst.params
                                filter_params_out = {'filter_class': filter_inst.__class__.__name__}
                                for key, value in params.items():
                                    filter_params_out[key] = value.value_type(value.value)
                                param_value_out.append(filter_params_out)
                        if len(param_value_out) > 0:
                            params_dict[param_name] = param_value_out
                    else:
                        param_value_out = param_value
                        params_dict[param_name] = param_value_out

                    break

        return params_dict


@pydantic.dataclasses.dataclass(config=pydantic.dataclasses.ConfigDict(arbitrary_types_allowed=True))
class ImageFile:
    img: torch.Tensor
    file_path: pathlib.Path
    img_read_mode: io.ImageReadMode


@pydantic.dataclasses.dataclass(config=pydantic.dataclasses.ConfigDict(arbitrary_types_allowed=True))
class RenderResult:
    # autopep8: off
    input_img                 : torch.Tensor | None = None
    input_axial_psd_h         : np.ndarray   | None = None
    input_axial_psd_v         : np.ndarray   | None = None
    input_psd_img             : np.ndarray   | None = None
    input_rad_psd             : np.ndarray   | None = None

    filter_img                : np.ndarray   | None = None
    filter_response_h         : np.ndarray   | None = None
    filter_response_v         : np.ndarray   | None = None

    filtered_axial_psd_h      : np.ndarray   | None = None
    filtered_axial_psd_v      : np.ndarray   | None = None
    filtered_psd_img          : np.ndarray   | None = None
    filtered_rad_psd          : np.ndarray   | None = None

    output_img                : torch.Tensor | None = None

    diff_img                  : torch.Tensor | None = None
    diff_min                  : float               = 0.0
    diff_max                  : float               = 0.0
    # autopep8: on


# --------------------------------------------------------------------------------------------------------------------------------------------------------
# Viewer


class FIRVisualizer(pyviewer_extended.MultiTexturesDockingViewer):
    """Visualizer class for finite impulse response filter designs
    """

    # ------------------------------------------------------------------------------------
    # Constants

    # autopep8: off
    KEY_INPUT         : typing.Final[str] = 'Input'
    KEY_OUTPUT        : typing.Final[str] = 'Output'
    KEY_DIFFERENCE    : typing.Final[str] = 'Difference'

    KEYS                                  = [KEY_INPUT, KEY_OUTPUT, KEY_DIFFERENCE]

    MIN_IMG_SIZE      : int               = 5    # it has to larger than 4
    MAX_IMG_SIZE      : int               = 2048 # 256 is already heavy with padding even if using CUDA backend ...

    PARAMS_CACHE_PATH : pathlib.Path      = pathlib.Path('.cache/fir_viz_params.json')

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
            full_screen_mode=multi_textures_viewer.hello_imgui.FullScreenMode.full_monitor_work_area,
            with_font_awesome=True,
            with_implot=True
        )

    @typing_extensions.override
    def setup_state(self) -> None:
        """Initialize the state of the visualizer. Called by the super class.
        """

        self.state.params = Params()
        self.state.render_result_cache = cachetools.FIFOCache(maxsize=8)
        self.state.cur_render_result = RenderResult()

        self.state.current_filter_item = 0
        self.state.current_filter_to_add_item = 0

        self.state.params_cache_file = self.PARAMS_CACHE_PATH

        # Load params
        if self.cache_params_to_file and self.PARAMS_CACHE_PATH.is_file():
            self.load_param_caches(self.PARAMS_CACHE_PATH)

        # Update param cache
        self.cached_params[self.cur_cache_param_id] = self.state.params

    def load_param_caches(self, file_path: pathlib.Path) -> None:
        try:
            with open(file_path, mode='r') as file:
                params = json.load(file)
        except Exception as e:
            params = None

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

    @property
    def filter_cmap(self) -> str:
        if self.params is not None:
            return MPL_CMAPS[self.params.filter_cmap_id]

        return 'viridis'

    @typing_extensions.override
    def compute(self) -> dict[str, np.ndarray]:
        # Check if the parameters have changed

        params_hash = hash(json.dumps(self.params.dump()))  # volatile hash

        if params_hash in self.state.render_result_cache:
            cached: RenderResult = self.state.render_result_cache[params_hash]
            self.state.cur_render_result = cached
            return {
                self.KEY_INPUT: cached.input_img,
                self.KEY_OUTPUT: cached.output_img,
                self.KEY_DIFFERENCE: cached.diff_img,
            }

        params = copy.deepcopy(self.params)  # Detach parameters
        render_result = RenderResult()

        # ---------------------------------------------------------------------------------------------------
        # Compute a sinusoidal image

        wave_number = params.wave_number
        target_freq = wave_number / params.img_size  # [cycles/pixel]

        if (params.img_mode == ImageMode.file or params.img_mode == ImageMode.file_RGB) and params.img_path.is_file():
            # From file
            img_read_mode = io.ImageReadMode.RGB if params.img_mode == ImageMode.file_RGB else io.ImageReadMode.GRAY

            # Buffer the image data to avoid reading it from file multiple times
            if self.base_img is None or self.base_img.file_path != params.img_path or self.base_img.img_read_mode != img_read_mode:
                img_decoded = io.decode_image(params.img_path, mode=img_read_mode)
                img_decoded = F.to_dtype(img_decoded, scale=True).to(dtype=_dtype, device=_device)
                self.base_img = ImageFile(img=img_decoded, file_path=params.img_path, img_read_mode=img_read_mode)

            img = self.base_img.img  # [channels, img_size, img_size]

            # Rotate the image
            img = F.rotate_image(
                img,
                angle=-params.rotate,
                center=None,  # Center the image
                interpolation=F.InterpolationMode.BILINEAR,
                expand=False,  # Do not expand the image
            )
        elif (params.img_mode == ImageMode.gaussian or params.img_mode == ImageMode.gaussian_RGB):
            num_channels = 3 if params.img_mode == ImageMode.gaussian_RGB else 1

            generator = torch.Generator(device=_device).manual_seed(params.seed)
            img = params.sigma * torch.randn((num_channels, params.img_size, params.img_size), dtype=_dtype, device=_device, generator=generator)  # zero DC
        else:
            canvas_size = params.img_size

            t = torch.linspace(0.0, canvas_size - 1.0, canvas_size, dtype=_dtype, device=_device)
            t = t - t.mean()  # Center the grid around zero
            grid = torch.stack(torch.meshgrid(t, t, indexing='ij'), dim=-1)  # [n_pixels, n_pixels, 2]

            rad_angle = math.pi * params.rotate / 180.0
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

        # Center crop the image
        img = F.center_crop(img, (params.img_size, params.img_size))

        # Images for visualization
        input_img = img.detach().clone().to(dtype=torch.float32, device=_device)  # [c, h, w]
        input_img = util.normalize_0_to_1(input_img, based_on_min_max=True)

        if img.shape[0] == 3:
            input_img = input_img.permute(1, 2, 0)
        elif self.img_cmap == 'gray':
            input_img = input_img.tile(3, 1, 1).permute(1, 2, 0)
        else:
            input_img = radpsd.apply_color_map(input_img, self.img_cmap).squeeze(0)

        # Sanity check
        assert input_img.ndim == 3
        assert input_img.shape[-1] == 3
        render_result.input_img = input_img

        # ---------------------------------------------------------------------------------------------------
        # FFT

        spectrum: torch.Tensor = torch.fft.fftn(img, dim=(-2, -1))
        spectrum = torch.fft.fftshift(spectrum, dim=(-2, -1))

        input_psd = spectrum.abs().square() / (img.shape[-2] * img.shape[-1])

        # Axial psd
        half_h = img.shape[-2] // 2
        half_w = img.shape[-1] // 2
        render_result.input_axial_psd_h = np.ascontiguousarray(np.fft.ifftshift(input_psd[:, half_h, :].cpu().numpy(), axes=1)[:, :half_w], dtype=np.float64)  # [C, W]
        render_result.input_axial_psd_v = np.ascontiguousarray(np.fft.ifftshift(input_psd[:, :, half_w].cpu().numpy(), axes=1)[:, :half_h], dtype=np.float64)  # [C, H]

        # 2D psd plot
        input_psd_img = input_psd.mean(dim=0)                      # take average on channel axis
        input_psd_img = 10.0 * torch.log10(input_psd_img + 1e-10)  # to decibels

        # Sanity check
        assert input_psd_img.ndim == 2
        render_result.input_psd_img = np.ascontiguousarray(input_psd_img.cpu().numpy()).astype(np.float32)

        # Compute the radial power spectral density
        input_rad_psd = _module.calc_radial_psd(
            input_psd.permute(1, 2, 0).unsqueeze(0).contiguous(),  # [1, H, W, C]
            _n_radial_divs,
            _n_polar_divs,
        )  # [1, n_divs, n_points, C]

        input_rad_psd = input_rad_psd.mean(dim=(0, 1))  # [n_points, C]
        render_result.input_rad_psd = input_rad_psd.cpu().numpy().astype(np.float64)

        # ---------------------------------------------------------------------------------------------------
        # Filter

        filter_prod = None
        for filter_inst in params.filters:
            f = filter_inst.compute_filter(size=spectrum.shape[-1], dtype=_dtype, device=_device)
            filter_prod = f if filter_prod is None else filter_prod * f

        if params.retain_power:
            filter_prod *= filter_prod.square().sum().rsqrt()

        # Apply filter
        if filter_prod is not None:
            spectrum *= filter_prod.unsqueeze(0)

            filter_img = np.ascontiguousarray(filter_prod.cpu().numpy(), dtype=np.float64)
        else:
            # Dummy
            filter_img = np.ones(shape=(spectrum.shape[-2], spectrum.shape[-1]), dtype=np.float64)

        # Sanity check
        assert filter_img.ndim == 2
        render_result.filter_img = 20.0 * np.log10(filter_img + 1e-15)  # to decibels
        render_result.filter_response_h = np.ascontiguousarray(np.fft.ifftshift(filter_img[half_h, :])[:half_w], dtype=np.float64)
        render_result.filter_response_v = np.ascontiguousarray(np.fft.ifftshift(filter_img[:, half_w])[:half_h], dtype=np.float64)

        # Filterd psd
        filtered_psd = spectrum.abs().square() / (img.shape[-2] * img.shape[-1])
        render_result.filtered_axial_psd_h = np.ascontiguousarray(np.fft.ifftshift(filtered_psd[:, half_h, :].cpu().numpy(), axes=1)[:, :half_w], dtype=np.float64)  # [C, W]
        render_result.filtered_axial_psd_v = np.ascontiguousarray(np.fft.ifftshift(filtered_psd[:, :, half_w].cpu().numpy(), axes=1)[:, :half_h], dtype=np.float64)  # [C, H]

        # 2D psd plot
        filtered_psd_img = filtered_psd.mean(dim=0)                      # take average on channel axis
        filtered_psd_img = 10.0 * torch.log10(filtered_psd_img + 1e-10)  # to decibels

        # Sanity check
        assert filtered_psd_img.ndim == 2
        render_result.filtered_psd_img = np.ascontiguousarray(filtered_psd_img.cpu().numpy()).astype(np.float32)

        # Compute the radial power spectral density
        filtered_rad_psd = _module.calc_radial_psd(
            filtered_psd.permute(1, 2, 0).unsqueeze(0).contiguous(),  # [1, H, W, C]
            _n_radial_divs,
            _n_polar_divs,
        )  # [1, n_divs, n_points, C]

        filtered_rad_psd = filtered_rad_psd.mean(dim=(0, 1))  # [n_points, C]
        render_result.filtered_rad_psd = filtered_rad_psd.cpu().numpy().astype(np.float64)

        # ---------------------------------------------------------------------------------------------------
        # Inverse FFT

        spectrum = torch.fft.ifftshift(spectrum, dim=(-2, -1))
        output: torch.Tensor = torch.fft.ifftn(spectrum, dim=(-2, -1)).real  # Only real part

        # Images for visualization
        output_img = output.detach().clone().to(dtype=torch.float32, device=_device)  # [c, h, w]
        output_img = util.normalize_0_to_1(output_img, based_on_min_max=True)

        if output_img.shape[0] == 3:
            output_img = output_img.permute(1, 2, 0)
        elif self.img_cmap == 'gray':
            output_img = output_img.tile(3, 1, 1).permute(1, 2, 0)
        else:
            output_img = radpsd.apply_color_map(output_img, self.img_cmap).squeeze(0)

        # Sanity check
        assert output_img.ndim == 3
        assert output_img.shape[-1] == 3
        render_result.output_img = output_img

        # ---------------------------------------------------------------------------------------------------
        # Diff

        diff = (output - img).abs()

        # Images for visualization
        diff_img = diff.detach().clone().to(dtype=torch.float32, device=_device)  # [c, h, w]
        diff_img = util.normalize_0_to_1(diff_img, based_on_min_max=True)

        if diff_img.shape[0] == 3:
            diff_img = diff_img.permute(1, 2, 0)
        elif self.img_cmap == 'gray':
            diff_img = diff_img.tile(3, 1, 1).permute(1, 2, 0)
        else:
            diff_img = radpsd.apply_color_map(diff_img, self.img_cmap).squeeze(0)

        # Sanity check
        assert diff_img.ndim == 3
        assert diff_img.shape[-1] == 3
        render_result.diff_img = diff_img
        render_result.diff_min = float(diff.min())
        render_result.diff_max = float(diff.max())

        # ---------------------------------------------------------------------------------------------------
        # Cache result

        self.state.render_result_cache[params_hash] = render_result
        self.state.cur_render_result = render_result

        return {
            self.KEY_INPUT: render_result.input_img,
            self.KEY_OUTPUT: render_result.output_img,
            self.KEY_DIFFERENCE: render_result.diff_img,
        }

    # ---------------------------------------------------------------------------------------------------
    # Dialog openers

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

    # ---------------------------------------------------------------------------------------------------
    # UI Elements

    @pyviewer_extended.dockable
    def toolbar(self) -> None:
        """Build the toolbar UI.
        """

        render_result: RenderResult = self.state.cur_render_result

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

        if imgui.button('Duplicate to next', size=(-1, 0)):
            next_cache_param_id = (self.cur_cache_param_id + 1) % self.NUM_PARAMS_CACHES
            self.cached_params[next_cache_param_id] = copy.deepcopy(self.cached_params[self.cur_cache_param_id])
            self.cur_cache_param_id = next_cache_param_id

        imgui.text('Tips: switch them with "arrow" or "number" keys')

        # Restore from cache
        self.state.params = self.cached_params[self.cur_cache_param_id]

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

        # ---------------------------------------------------------------------------------------------------
        # Filter parameters
        imgui.separator()
        if imgui.collapsing_header('Filter', flags=imgui.TreeNodeFlags_.default_open):
            # Add
            imgui.separator_text('Add filter')
            self.state.current_filter_to_add_item = imgui.combo('##Filter to add', self.state.current_filter_to_add_item, filtering.filter_names)[1]
            if imgui.button('Add', size=(-1, 0)):
                filter_name = filtering.filter_names[self.state.current_filter_to_add_item]
                filter_inst = filtering.filters[filter_name]()
                self.params.filters.append(filter_inst)
                self.state.current_filter_item = len(self.params.filters) - 1  # Move cursor to the last

            # Remove
            imgui.separator_text('Remove filter')
            if imgui.button('Remove', size=(-1, 0)) and self.state.current_filter_item < len(self.params.filters):
                self.params.filters.pop(self.state.current_filter_item)

            # List
            imgui.separator_text('Filter list')
            filter_names = [filtering.filter_class_to_name[f.__class__] for f in self.params.filters]
            self.state.current_filter_item = imgui.list_box('Filters', self.state.current_filter_item, filter_names)[1]

            # Params
            imgui.separator_text('Filter params')
            if self.state.current_filter_item < len(self.params.filters):
                current_filter = self.params.filters[self.state.current_filter_item]
                current_filter.draw_param_widgets()

            self.params.retain_power = imgui.checkbox('Retain Gain', self.params.retain_power)[1]

        # ---------------------------------------------------------------------------------------------------
        # Visualization parameters

        imgui.separator()
        if imgui.collapsing_header('Visualization'):
            imgui.separator_text('Color maps')
            self.params.img_cmap_id = imgui.combo('Image color map', self.params.img_cmap_id, MPL_CMAPS)[1]
            self.params.psd_cmap_id = imgui.combo('PSD color map', self.params.psd_cmap_id, MPL_CMAPS)[1]
            self.params.filter_cmap_id = imgui.combo('Filter color map', self.params.filter_cmap_id, MPL_CMAPS)[1]

            imgui.separator_text('Power Spectral Density Profile of Input Image')
            # autopep8: off
            imgui.push_id('Power Spectral Density Profile of Input Image (Fix y-limits)'); self.params.input_psd_profile_ylim_fixed = imgui.checkbox('Fix y-limits', self.params.input_psd_profile_ylim_fixed)[1]; imgui.pop_id()
            if self.params.input_psd_profile_ylim_fixed:
                imgui.push_id('Power Spectral Density Profile of Input Image  (Min y-limit)'); self.params.input_psd_profile_ylim_min = imgui.input_float('Min y-limit', self.params.input_psd_profile_ylim_min, step=1e-10, format='%.2e')[1]; imgui.pop_id()
                imgui.push_id('Power Spectral Density Profile of Input Image  (Max y-limit)'); self.params.input_psd_profile_ylim_max = imgui.input_float('Max y-limit', self.params.input_psd_profile_ylim_max, step=1e-10, format='%.2e')[1]; imgui.pop_id()
            imgui.push_id('Power Spectral Density Profile of Input Image  (X in log scale)'); self.params.input_psd_profile_xscale_log = imgui.checkbox('X in log scale', self.params.input_psd_profile_xscale_log)[1]; imgui.pop_id()
            imgui.push_id('Power Spectral Density Profile of Input Image  (Y in log scale)'); self.params.input_psd_profile_yscale_log = imgui.checkbox('Y in log scale', self.params.input_psd_profile_yscale_log)[1]; imgui.pop_id()
            # autopep8: on

            imgui.separator_text('Filter Response')
            # autopep8: off
            imgui.push_id('Filter Response (Fix y-limits)'); self.params.filter_response_ylim_fixed = imgui.checkbox('Fix y-limits', self.params.filter_response_ylim_fixed)[1]; imgui.pop_id()
            if self.params.filter_response_ylim_fixed:
                imgui.push_id('Filter Response (Min y-limit)'); self.params.filter_response_ylim_min = imgui.input_float('Min y-limit', self.params.filter_response_ylim_min, step=1e-10, format='%.2e')[1]; imgui.pop_id()
                imgui.push_id('Filter Response (Max y-limit)'); self.params.filter_response_ylim_max = imgui.input_float('Max y-limit', self.params.filter_response_ylim_max, step=1e-10, format='%.2e')[1]; imgui.pop_id()
            imgui.push_id('Filter Response (X in log scale)'); self.params.filter_response_xscale_log = imgui.checkbox('X in log scale', self.params.filter_response_xscale_log)[1]; imgui.pop_id()
            # autopep8: on

            imgui.separator_text('Power Spectral Density Profile of Filtered Image')
            # autopep8: off
            imgui.push_id('Power Spectral Density Profile of Filtered Image (Fix y-limits)'); self.params.filtered_psd_profile_ylim_fixed = imgui.checkbox('Fix y-limits', self.params.filtered_psd_profile_ylim_fixed)[1]; imgui.pop_id()
            if self.params.filtered_psd_profile_ylim_fixed:
                imgui.push_id('Power Spectral Density Profile of Filtered Image (Min y-limit)'); self.params.filtered_psd_profile_ylim_min = imgui.input_float('Min y-limit', self.params.filtered_psd_profile_ylim_min, step=1e-10, format='%.2e')[1]; imgui.pop_id()
                imgui.push_id('Power Spectral Density Profile of Filtered Image (Max y-limit)'); self.params.filtered_psd_profile_ylim_max = imgui.input_float('Max y-limit', self.params.filtered_psd_profile_ylim_max, step=1e-10, format='%.2e')[1]; imgui.pop_id()
            imgui.push_id('Power Spectral Density Profile of Filtered Image (X in log scale)'); self.params.filtered_psd_profile_xscale_log = imgui.checkbox('X in log scale', self.params.filtered_psd_profile_xscale_log)[1]; imgui.pop_id()
            imgui.push_id('Power Spectral Density Profile of Filtered Image (Y in log scale)'); self.params.filtered_psd_profile_yscale_log = imgui.checkbox('Y in log scale', self.params.filtered_psd_profile_yscale_log)[1]; imgui.pop_id()
            # autopep8: on

        # ---------------------------------------------------------------------------------------------------
        # Note
        imgui.separator()
        if imgui.collapsing_header('Note', flags=imgui.TreeNodeFlags_.default_open):
            imgui.text(f'Difference: (Min, Max) = ({render_result.diff_min}, {render_result.diff_max})')

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

    @pyviewer_extended.dockable
    def input_psd_plot(self) -> None:
        render_result: RenderResult = self.state.cur_render_result

        # ---------------------------------------------------------------------------------------------------
        # Plot the power spectral density

        x_avail, _ = imgui.get_content_region_avail()
        color_bar_prop = 0.10  # 10% for the color bar
        plot_width = x_avail * (1.0 - color_bar_prop)
        color_bar_width = x_avail - plot_width

        cmap = getattr(implot.Colormap_, self.psd_cmap, None)
        if cmap is not None:
            implot.push_colormap(cmap.value)

        if render_result.input_psd_img is not None:
            psd_img = render_result.input_psd_img
            scale_min = np.min(psd_img)
            scale_max = np.max(psd_img)

            if implot.begin_plot(
                'Power Spectral Density of Input Image',
                size=(plot_width, -1),
                flags=implot.Flags_.no_legend.value | implot.Flags_.equal.value
            ):
                half_size = self.params.img_size // 2

                implot.setup_axes("Horizontal Frequency [wave number]", "Vertical Frequency [wave number]")
                implot.setup_axes_limits(-half_size, half_size, -half_size, half_size)
                implot.plot_heatmap(
                    label_id='##Heatmap Power Spectral Density of Input Image',
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
    def input_psd_profile_plot(self):
        render_result: RenderResult = self.state.cur_render_result

        if imgui.begin_tab_bar('Power Spectral Density Profiles of Input Image'):
            # ----------------------------------------------------------------------------------------------------------------------------------------------------
            # Plot the radial power spectral density

            if imgui.begin_tab_item_simple('Radial'):
                if render_result.input_rad_psd is not None and implot.begin_plot(
                    'Radial Power Spectral Density of Input Image',
                    size=(-1, -1),
                    flags=implot.Flags_.no_legend.value if render_result.input_rad_psd.shape[1] == 1 else 0
                ):
                    rad_psd: np.ndarray = render_result.input_rad_psd
                    is_db_scale = self.params.input_psd_profile_yscale_log
                    if is_db_scale:
                        rad_psd = rad_psd.copy() ** 10

                    # Setup x axis
                    implot.setup_axis(implot.ImAxis_.x1, "Frequency [wave number]")
                    implot.setup_axis_limits(implot.ImAxis_.x1, 1.0, self.params.img_size / 2, imgui.Cond_.always.value)

                    # Setup y axis
                    implot.setup_axis(implot.ImAxis_.y1, "Power Spectral Density" + (' [dB]' if is_db_scale else ''), flags=implot.AxisFlags_.auto_fit.value)
                    if self.params.input_psd_profile_ylim_fixed:
                        implot.setup_axis_limits(implot.ImAxis_.y1, self.params.input_psd_profile_ylim_min, self.params.input_psd_profile_ylim_max, imgui.Cond_.always.value)

                    # Set log-log scale
                    implot.setup_axis_scale(implot.ImAxis_.x1, implot.Scale_.log10.value if self.params.input_psd_profile_xscale_log else implot.Scale_.linear.value)
                    implot.setup_axis_scale(implot.ImAxis_.y1, implot.Scale_.log10.value if is_db_scale else implot.Scale_.linear.value)

                    freq = radpsd.radial_freq(img_size=self.params.img_size, n_radial_bins=len(rad_psd), dtype=np.float64)  # [cycles/pix]
                    freq = freq * self.params.img_size  # ranged in [0, ..., img_size / 2]
                    freq = np.ascontiguousarray(freq[1:])  # w/o DC

                    if rad_psd.shape[1] == 3:
                        # RGB
                        implot.set_next_line_style(imgui.ImVec4(1.0, 0.0, 0.0, 1.0))
                        util.make_unique(implot.plot_line, 'Radial PSD - Red', freq, np.ascontiguousarray(rad_psd[1:, 0]))
                        implot.set_next_line_style(imgui.ImVec4(0.0, 1.0, 0.0, 1.0))
                        util.make_unique(implot.plot_line, 'Radial PSD - Green', freq, np.ascontiguousarray(rad_psd[1:, 1]))
                        implot.set_next_line_style(imgui.ImVec4(0.0, 0.0, 1.0, 1.0))
                        util.make_unique(implot.plot_line, 'Radial PSD - Blue', freq, np.ascontiguousarray(rad_psd[1:, 2]))
                    else:
                        util.make_unique(implot.plot_line, 'Radial Power Spectral Density', freq, np.ascontiguousarray(rad_psd[1:, 0]))

                    implot.end_plot()
                imgui.end_tab_item()

            # ----------------------------------------------------------------------------------------------------------------------------------------------------
            # Plot the axial power spectral density

            for label, axial_psd in (('Horizontal', render_result.input_axial_psd_h), ('Vertical', render_result.input_axial_psd_v)):
                if imgui.begin_tab_item_simple(label):
                    # Plot the axial power spectral density
                    if axial_psd is not None and render_result.input_psd_img is not None and implot.begin_plot(
                        f'{label} Power Spectral Density of Input Image',
                        size=(-1, -1),
                        flags=implot.Flags_.no_legend.value if axial_psd.shape[0] == 1 else 0
                    ):
                        is_db_scale = self.params.input_psd_profile_yscale_log
                        if is_db_scale:
                            axial_psd = axial_psd.copy() ** 10

                        padded_img_size = render_result.input_psd_img.shape[-1]
                        freq = np.fft.fftfreq(padded_img_size, d=1.0/self.params.img_size)[1:axial_psd.shape[1]].astype(np.float64)  # assume that the img is square

                        # Setup x axis
                        implot.setup_axis(implot.ImAxis_.x1, "Frequency [wave number]")
                        implot.setup_axis_limits(implot.ImAxis_.x1, freq.min(), freq.max(), imgui.Cond_.always.value)

                        # Setup y axis
                        implot.setup_axis(implot.ImAxis_.y1, "Power Spectral Density" + (' [dB]' if is_db_scale else ''), flags=implot.AxisFlags_.auto_fit.value)
                        if self.params.input_psd_profile_ylim_fixed:
                            implot.setup_axis_limits(implot.ImAxis_.y1, self.params.input_psd_profile_ylim_min, self.params.input_psd_profile_ylim_max, imgui.Cond_.always.value)

                        # Set log-log scale
                        implot.setup_axis_scale(implot.ImAxis_.x1, implot.Scale_.log10.value if self.params.input_psd_profile_xscale_log else implot.Scale_.linear.value)
                        implot.setup_axis_scale(implot.ImAxis_.y1, implot.Scale_.log10.value if is_db_scale else implot.Scale_.linear.value)

                        if axial_psd.shape[0] == 3:
                            # RGB
                            implot.set_next_line_style(imgui.ImVec4(1.0, 0.0, 0.0, 1.0))
                            util.make_unique(implot.plot_line, f'{label} PSD - Red', freq, np.ascontiguousarray(axial_psd[0, 1:]))
                            implot.set_next_line_style(imgui.ImVec4(0.0, 1.0, 0.0, 1.0))
                            util.make_unique(implot.plot_line, f'{label} PSD - Green', freq, np.ascontiguousarray(axial_psd[1, 1:]))
                            implot.set_next_line_style(imgui.ImVec4(0.0, 0.0, 1.0, 1.0))
                            util.make_unique(implot.plot_line, f'{label} PSD - Blue', freq, np.ascontiguousarray(axial_psd[2, 1:]))
                        else:
                            util.make_unique(implot.plot_line, f'{label} Power Spectral Density', freq, np.ascontiguousarray(axial_psd[0, 1:]))

                        implot.end_plot()
                    imgui.end_tab_item()
            imgui.end_tab_bar()

    @pyviewer_extended.dockable
    def filter_plot(self) -> None:
        render_result: RenderResult = self.state.cur_render_result

        # ---------------------------------------------------------------------------------------------------
        # Plot the filter

        x_avail, _ = imgui.get_content_region_avail()
        color_bar_prop = 0.10  # 10% for the color bar
        plot_width = x_avail * (1.0 - color_bar_prop)
        color_bar_width = x_avail - plot_width

        cmap = getattr(implot.Colormap_, self.filter_cmap, None)
        if cmap is not None:
            implot.push_colormap(cmap.value)

        if render_result.filter_img is not None:
            filter_img = render_result.filter_img
            scale_min = np.min(filter_img)
            scale_max = np.max(filter_img)

            if implot.begin_plot(
                'Filter',
                size=(plot_width, -1),
                flags=implot.Flags_.no_legend.value | implot.Flags_.equal.value
            ):

                half_size = self.params.img_size // 2

                implot.setup_axes("Horizontal Frequency [wave number]", "Vertical Frequency [wave number]")
                implot.setup_axes_limits(-half_size, half_size, -half_size, half_size)
                util.make_unique(
                    implot.plot_heatmap,
                    '##Heatmap Filter',
                    values=filter_img,
                    scale_min=scale_min,
                    scale_max=scale_max,
                    bounds_min=implot.Point(-half_size, -half_size),
                    bounds_max=implot.Point(half_size, half_size),
                    label_fmt='',
                )

                implot.end_plot()

            imgui.same_line()
            util.make_unique(implot.colormap_scale, "Filter gain [dB]", scale_min, scale_max, size=(color_bar_width, -1))

        if cmap is not None:
            implot.pop_colormap()

    @pyviewer_extended.dockable
    def filter_response_plot(self):
        render_result: RenderResult = self.state.cur_render_result

        if imgui.begin_tab_bar('Filter Responses'):
            for label, filter_response in (('Horizontal', render_result.filter_response_h), ('Vertical', render_result.filter_response_v)):
                if imgui.begin_tab_item_simple(label):
                    if filter_response is not None and render_result.input_psd_img is not None and implot.begin_plot(
                        f'{label} Filter Response',
                        size=(-1, -1),
                        flags=implot.Flags_.no_legend.value if filter_response.shape[0] == 1 else 0
                    ):
                        filter_response = np.log10(filter_response.copy() ** 20 + 1e-15)  # NOTE: dB for not power but amplitude: 20 * log10(psd)

                        padded_img_size = render_result.input_psd_img.shape[-1]
                        freq = np.fft.fftfreq(padded_img_size, d=1.0/self.params.img_size)[1:len(filter_response)].astype(np.float64)  # assume that the img is square

                        # Setup x axis
                        implot.setup_axis(implot.ImAxis_.x1, "Frequency [wave number]")
                        implot.setup_axis_limits(implot.ImAxis_.x1, freq.min(), freq.max(), imgui.Cond_.always.value)
                        if self.params.filter_response_ylim_fixed:
                            implot.setup_axis_limits(implot.ImAxis_.y1, self.params.filter_response_ylim_min, self.params.filter_response_ylim_max, imgui.Cond_.always.value)

                        # Setup y axis
                        implot.setup_axis(implot.ImAxis_.y1, "Gain [dB]", flags=implot.AxisFlags_.auto_fit.value)

                        # Set log-log scale
                        implot.setup_axis_scale(implot.ImAxis_.x1, implot.Scale_.log10.value if self.params.filter_response_xscale_log else implot.Scale_.linear.value)
                        implot.setup_axis_scale(implot.ImAxis_.y1, implot.Scale_.linear.value)

                        util.make_unique(implot.plot_line, f'{label} Filter Response', freq, np.ascontiguousarray(filter_response[1:]))

                        implot.end_plot()
                    imgui.end_tab_item()
            imgui.end_tab_bar()

    @pyviewer_extended.dockable
    def filtered_psd_plot(self) -> None:
        render_result: RenderResult = self.state.cur_render_result

        x_avail, _ = imgui.get_content_region_avail()
        color_bar_prop = 0.10  # 10% for the color bar
        plot_width = x_avail * (1.0 - color_bar_prop)
        color_bar_width = x_avail - plot_width

        cmap = getattr(implot.Colormap_, self.psd_cmap, None)
        if cmap is not None:
            implot.push_colormap(cmap.value)

        if render_result.filtered_psd_img is not None:
            psd_img = render_result.filtered_psd_img
            scale_min = np.min(psd_img)
            scale_max = np.max(psd_img)

            if implot.begin_plot(
                'Power Spectral Density of Filtered Image',
                size=(plot_width, -1),
                flags=implot.Flags_.no_legend.value | implot.Flags_.equal.value
            ):

                half_size = self.params.img_size // 2

                implot.setup_axes("Horizontal Frequency [wave number]", "Vertical Frequency [wave number]")
                implot.setup_axes_limits(-half_size, half_size, -half_size, half_size)
                util.make_unique(
                    implot.plot_heatmap,
                    '##Heatmap Power Spectral Density of Filtered Image',
                    values=psd_img,
                    scale_min=scale_min,
                    scale_max=scale_max,
                    bounds_min=implot.Point(-half_size, -half_size),
                    bounds_max=implot.Point(half_size, half_size),
                    label_fmt='',
                )

                implot.end_plot()

            imgui.same_line()
            util.make_unique(implot.colormap_scale, "Power Spectral Density [dB]", scale_min, scale_max, size=(color_bar_width, -1))

        if cmap is not None:
            implot.pop_colormap()

    @pyviewer_extended.dockable
    def filtered_psd_profile_plot(self):
        render_result: RenderResult = self.state.cur_render_result

        if imgui.begin_tab_bar('Power Spectral Density Profiles of Filtered Image'):
            # ----------------------------------------------------------------------------------------------------------------------------------------------------
            # Plot the radial power spectral density

            if imgui.begin_tab_item_simple('Radial'):
                if render_result.filtered_rad_psd is not None and implot.begin_plot(
                    'Radial Power Spectral Density of Filtered Image',
                    size=(-1, -1),
                    flags=implot.Flags_.no_legend.value if render_result.filtered_rad_psd.shape[1] == 1 else 0
                ):
                    rad_psd: np.ndarray = render_result.filtered_rad_psd
                    is_db_scale = self.params.filtered_psd_profile_yscale_log
                    if is_db_scale:
                        rad_psd = rad_psd.copy() ** 10

                    # Setup x axis
                    implot.setup_axis(implot.ImAxis_.x1, "Frequency [wave number]")
                    implot.setup_axis_limits(implot.ImAxis_.x1, 1.0, self.params.img_size / 2, imgui.Cond_.always.value)

                    # Setup y axis
                    implot.setup_axis(implot.ImAxis_.y1, "Power Spectral Density" + (" [dB]"if is_db_scale else ""), flags=implot.AxisFlags_.auto_fit.value)
                    if self.params.filtered_psd_profile_ylim_fixed:
                        implot.setup_axis_limits(implot.ImAxis_.y1, self.params.filtered_psd_profile_ylim_min, self.params.filtered_psd_profile_ylim_max, imgui.Cond_.always.value)

                    # Set log-log scale
                    implot.setup_axis_scale(implot.ImAxis_.x1, implot.Scale_.log10.value if self.params.filtered_psd_profile_xscale_log else implot.Scale_.linear.value)
                    implot.setup_axis_scale(implot.ImAxis_.y1, implot.Scale_.log10.value if is_db_scale else implot.Scale_.linear.value)

                    freq = radpsd.radial_freq(img_size=self.params.img_size, n_radial_bins=len(rad_psd), dtype=np.float64)  # [cycles/pix]
                    freq = freq * self.params.img_size  # ranged in [0, ..., img_size / 2]
                    freq = np.ascontiguousarray(freq[1:])  # w/o DC

                    if rad_psd.shape[1] == 3:
                        # RGB
                        implot.set_next_line_style(imgui.ImVec4(1.0, 0.0, 0.0, 1.0))
                        util.make_unique(implot.plot_line, 'Radial PSD - Red', freq, np.ascontiguousarray(rad_psd[1:, 0]))
                        implot.set_next_line_style(imgui.ImVec4(0.0, 1.0, 0.0, 1.0))
                        util.make_unique(implot.plot_line, 'Radial PSD - Green', freq, np.ascontiguousarray(rad_psd[1:, 1]))
                        implot.set_next_line_style(imgui.ImVec4(0.0, 0.0, 1.0, 1.0))
                        util.make_unique(implot.plot_line, 'Radial PSD - Blue', freq, np.ascontiguousarray(rad_psd[1:, 2]))
                    else:
                        util.make_unique(implot.plot_line, 'Radial Power Spectral Density', freq, np.ascontiguousarray(rad_psd[1:, 0]))

                    implot.end_plot()
                imgui.end_tab_item()

            # ----------------------------------------------------------------------------------------------------------------------------------------------------
            # Plot the axial power spectral density

            for label, axial_psd in (('Horizontal', render_result.filtered_axial_psd_h), ('Vertical', render_result.filtered_axial_psd_v)):
                if imgui.begin_tab_item_simple(label):
                    # Plot the axial power spectral density
                    if axial_psd is not None and render_result.input_psd_img is not None and implot.begin_plot(
                        f'{label} Power Spectral Density of Filtered Image',
                        size=(-1, -1),
                        flags=implot.Flags_.no_legend.value if axial_psd.shape[0] == 1 else 0
                    ):
                        is_db_scale = self.params.filtered_psd_profile_yscale_log
                        if is_db_scale:
                            axial_psd = axial_psd.copy() ** 10

                        padded_img_size = render_result.input_psd_img.shape[-1]
                        freq = np.fft.fftfreq(padded_img_size, d=1.0/self.params.img_size)[1:axial_psd.shape[1]].astype(np.float64)  # assume that the img is square

                        # Setup x axis
                        implot.setup_axis(implot.ImAxis_.x1, "Frequency [wave number]")
                        implot.setup_axis_limits(implot.ImAxis_.x1, freq.min(), freq.max(), imgui.Cond_.always.value)

                        # Setup y axis
                        implot.setup_axis(implot.ImAxis_.y1, "Power Spectral Density" + (" [dB]"if is_db_scale else ""), flags=implot.AxisFlags_.auto_fit.value)
                        if self.params.filtered_psd_profile_ylim_fixed:
                            implot.setup_axis_limits(implot.ImAxis_.y1, self.params.filtered_psd_profile_ylim_min, self.params.filtered_sd_profile_ylim_max, imgui.Cond_.always.value)

                        # Set log-log scale
                        implot.setup_axis_scale(implot.ImAxis_.x1, implot.Scale_.log10.value if self.params.filtered_psd_profile_xscale_log else implot.Scale_.linear.value)
                        implot.setup_axis_scale(implot.ImAxis_.y1, implot.Scale_.log10.value if is_db_scale else implot.Scale_.linear.value)

                        if axial_psd.shape[0] == 3:
                            # RGB
                            implot.set_next_line_style(imgui.ImVec4(1.0, 0.0, 0.0, 1.0))
                            util.make_unique(implot.plot_line, f'{label} PSD - Red', freq, np.ascontiguousarray(axial_psd[0, 1:]))
                            implot.set_next_line_style(imgui.ImVec4(0.0, 1.0, 0.0, 1.0))
                            util.make_unique(implot.plot_line, f'{label} PSD - Green', freq, np.ascontiguousarray(axial_psd[1, 1:]))
                            implot.set_next_line_style(imgui.ImVec4(0.0, 0.0, 1.0, 1.0))
                            util.make_unique(implot.plot_line, f'{label} PSD - Blue', freq, np.ascontiguousarray(axial_psd[2, 1:]))
                        else:
                            util.make_unique(implot.plot_line, f'{label} Power Spectral Density', freq, np.ascontiguousarray(axial_psd[0, 1:]))

                        implot.end_plot()
                    imgui.end_tab_item()
            imgui.end_tab_bar()

    @typing_extensions.override
    def setup_docking_layout(self, layout_funcs):
        # Graceful initial layout ----------------------------------------------------------------------------------------------------------------
        # Window layout
        #   + ---------------------------- + ---------------------------- + ---------------------------- + ---------------------------- +
        #   | 'Input'                      | 'Difference'                 | 'Output'                     |                              |
        #   + ---------------------------- + ---------------------------- + ---------------------------- +                              |
        #   | 'input_psd_plot'             | 'filter_plot'                | 'filtered_psd_plot'          | 'toolbar'                    |
        #   + ---------------------------- + ---------------------------- + ---------------------------- +                              |
        #   | 'input_psd_profile_plot'     | 'filter_response_plot'       | 'filtered_psd_profile_plot'  |                              |
        #   + ---------------------------- + ---------------------------- + ---------------------------- + ---------------------------- +
        #
        # Dockspace names
        #   + ---------------------------- + ---------------------------- + ---------------------------- + ---------------------------- +
        #   | 'MainDockSpace'              | 'Dock01'                     | 'Dock02'                     |                              |
        #   + ---------------------------- + ---------------------------- + ---------------------------- +                              |
        #   | 'Dock10'                     | 'Dock11'                     | 'Dock12'                     | 'Dock03'                     |
        #   + ---------------------------- + ---------------------------- + ---------------------------- +                              |
        #   | 'Dock20'                     | 'Dock21'                     | 'Dock22'                     |                              |
        #   + ---------------------------- + ---------------------------- + ---------------------------- + ---------------------------- +

        splits = [
            # autopep8: off
            # Split colomns first
            multi_textures_viewer.hello_imgui.DockingSplit('MainDockSpace', 'Dock01', imgui.Dir.right, ratio_=3/4),
            multi_textures_viewer.hello_imgui.DockingSplit(       'Dock01', 'Dock02', imgui.Dir.right, ratio_=2/3),
            multi_textures_viewer.hello_imgui.DockingSplit(       'Dock02', 'Dock03', imgui.Dir.right, ratio_=1/2),

            # Next, along row
            multi_textures_viewer.hello_imgui.DockingSplit('MainDockSpace', 'Dock10',  imgui.Dir.down, ratio_=2/3),
            multi_textures_viewer.hello_imgui.DockingSplit(       'Dock10', 'Dock20',  imgui.Dir.down, ratio_=1/2),

            multi_textures_viewer.hello_imgui.DockingSplit(       'Dock01', 'Dock11',  imgui.Dir.down, ratio_=2/3),
            multi_textures_viewer.hello_imgui.DockingSplit(       'Dock11', 'Dock21',  imgui.Dir.down, ratio_=1/2),

            multi_textures_viewer.hello_imgui.DockingSplit(       'Dock02', 'Dock12',  imgui.Dir.down, ratio_=2/3),
            multi_textures_viewer.hello_imgui.DockingSplit(       'Dock12', 'Dock22',  imgui.Dir.down, ratio_=1/2),
            # autopep8: on
        ]

        title_to_dockspace_name = {
            # autopep8: off
                              'Input' : 'MainDockSpace',           'Difference': 'Dock01',                    'Output': 'Dock02',
                     'input_psd_plot' :        'Dock10',          'filter_plot': 'Dock11',         'filtered_psd_plot': 'Dock12', 'toolbar': 'Dock03',
             'input_psd_profile_plot' :        'Dock20', 'filter_response_plot': 'Dock21', 'filtered_psd_profile_plot': 'Dock22',
            # autopep8: on
        }

        windows = [multi_textures_viewer.hello_imgui.DockableWindow(f._title, title_to_dockspace_name[f._title], f, can_be_closed_=True) for f in layout_funcs]

        return splits, windows


# -------------------------------------------------------------------------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='A interactive visualization program for high-quality finite impulse response filter design',
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

    viz = FIRVisualizer('FIR Viz', enable_vsync=args.vsync, cache_params=not args.no_cache)
    logger.info('Bye!')

# -------------------------------------------------------------------------------------------------------------------------------------------------
