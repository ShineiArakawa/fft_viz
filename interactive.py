# autopep8: off
import copy
import enum
import math
import pathlib
import typing

import matplotlib as mpl

# Set Agg backend
mpl.use('Agg')


import cv2
import numpy as np
import pydantic.dataclasses as dataclasses
import pyviewer.docking_viewer as docking_viewer
import research_utilities as rutils
import research_utilities.signal as _signal
import research_utilities.torch_util as _torch_util
import torch
import torchvision.transforms.v2.functional as F
from imgui_bundle import imgui, implot

# autopep8: on

# --------------------------------------------------------------------------------------------------------------------------------------------------------
# Load the C++/CUDA module

# Check PyTorch is compiled with CUDA and nvcc is available
cuda_enabled = torch.cuda.is_available() and _torch_util.get_extension_loader()._check_command('nvcc')

# Build and load the C++/CUDA module to compute the radial power spectral density
print('Loading the C++/CUDA module...')
_module = _signal._get_cpp_module(is_cuda=cuda_enabled, with_omp=True)
print('done.')

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
    # autopep8: on

    def __str__(self) -> str:
        return self.value


MPL_CMAPS: typing.Final[list[str]] = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'gray']


@dataclasses.dataclass
class Params:
    """Parameters for the Color of Noise visualizer.
    """

    # autopep8: off
    img_mode                            : int          = int(ImageMode.sinusoidal)  # image mode
    img_path                            : pathlib.Path = pathlib.Path()             # image path
    img_size                            : int          = 256                        # image size
    enable_super_sampling               : bool         = False                      # enable super sampling
    super_sampling_factor               : int          = 1                          # super sampling factor
    enable_pre_filering                 : bool         = False                      # enable pre-filtering
    kernel_size                         : int          = 15                         # kernel size for pre-filtering
    kernel_sigma                        : float        = 0.3 * 6 + 0.8              # sigma for pre-filtering. See also: https://docs.pytorch.org/vision/main/generated/torchvision.transforms.functional.gaussian_blur.html
    wave_number                         : float        = 50.0                       # wave number
    rotate                              : float        = 0.0                        # rotation angle in degrees
    affine_interpolation_method         : int          = int(InterpMethod.nearest)  # interpolation method for the affine transformation

    apply_windowing                     : bool         = False                      # apply windowing to the image
    beta                                : float        = 8.0                        # beta parameter for the kaiser window
    apply_padding                       : bool         = False                      # apply zero padding to the image
    padding_factor                      : int          = 4                          # padding factor for the FFT

    img_cmap_id                         : int          = 5                          # color map ID for the sinusoidal image, by default 'gray'
    psd_cmap_id                         : int          = 1                          # color map ID for the power spectrum density, by default 'plasma'
    # autopep8: on


# --------------------------------------------------------------------------------------------------------------------------------------------------------
# utilities


def normalize_0_to_1(x: np.ndarray, based_on_min_max: bool = False) -> np.ndarray:
    """Normalize the input array to the range [0, 1].

    Parameters
    ----------
    x : np.ndarray
        Input array to be normalized.
    based_on_min_max : bool, optional
        If True, the normalization is based on the min and max values of the input array. If False, the normalization is based on the range [-1, 1]. Default is False.

    Returns
    -------
    np.ndarray
        Normalized array in the range [0, 1].

    Raises
    ------
    ValueError
        If the input array is not 2D or 3D.
    """

    if based_on_min_max:
        if x.ndim == 3:
            min = np.min(x, axis=(0, 1), keepdims=True)
            max = np.max(x, axis=(0, 1), keepdims=True)
        elif x.ndim == 2:
            min = np.min(x)
            max = np.max(x)
        else:
            raise ValueError("Input array must be 2D or 3D.")
    else:
        # [-1, 1] -> [0, 1]
        min = -1.0
        max = 1.0

    return np.clip((x - min) / (max - min + 1e-8), 0.0, 1.0)


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


class FFTVisualizer(docking_viewer.DockingViewer):
    """Visualizer class for FFT
    """

    def __init__(self, name):
        self.base_img: torch.Tensor | None = None

        # ------------------------------------------------------------------------------------
        # Set the device and data type based on the available hardware
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.dtype = torch.float64
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
            self.dtype = torch.float32  # MPS does not support float64
        else:
            self.device = torch.device('cpu')
            self.dtype = torch.float64

        # ------------------------------------------------------------------------------------
        super().__init__(name, with_font_awesome=True, with_implot=True)

    def setup_state(self) -> None:
        """Initialize the state of the visualizer. Called by the super class.
        """

        self.state.params = Params()
        self.state.prev_params = None

        self.state.img = None
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

    def compute(self) -> np.ndarray:
        # Check if the parameters have changed
        if self.state.prev_params is not None and self.state.prev_params == self.params:
            return self.state.img

        # ---------------------------------------------------------------------------------------------------
        # Compute a sinusoidal image

        wave_number = self.params.wave_number
        target_freq = wave_number / self.params.img_size  # [cycles/pixel]

        super_sampling_factor = self.params.super_sampling_factor if self.params.enable_super_sampling else 1

        if self.base_img is None or self.base_img.shape[0] != self.params.img_size:
            # if self.params.img_mode == ImageMode.file and self.params.img_path.is_file():
            #     img = cv2.imread(str(self.params.img_path), cv2.IMREAD_GRAYSCALE)
            #     img = cv2.resize(img, (int(self.params.img_size * 1.5), int(self.params.img_size * 1.5)), interpolation=cv2.INTER_LANCZOS4)
            #     img = cv2.flip(img, -1)
            #     img = torch.from_numpy(img).to(self.dtype).to(self.device)
            # else:

            canvas_size = math.floor(self.params.img_size * 1.5)  # Allocate larger canvas for the rotation
            t = torch.linspace(0.0, canvas_size - 1, canvas_size * super_sampling_factor, dtype=self.dtype, device=self.device)
            t = t - canvas_size // 2  # center the image

            x = torch.sin(2.0 * np.pi * target_freq * t)
            img = torch.tile(x, (len(x), 1))

            self.base_img = img.clone()
        else:
            img = self.base_img.clone()

        # Rotate the image
        img = F.affine(
            img.unsqueeze(0),
            angle=(self.params.rotate - 180.0),
            translate=(0.0, 0.0),
            scale=1.0,
            shear=(0.0, 0.0),
            interpolation=InterpMethod(self.params.affine_interpolation_method).to_torch()
        )

        # Center crop the image
        if self.params.enable_super_sampling and self.params.enable_pre_filering:
            # Prefilering
            img = F.gaussian_blur(
                img,
                kernel_size=(self.params.kernel_size, self.params.kernel_size),
                sigma=(self.params.kernel_sigma, self.params.kernel_sigma)
            )

        img = F.center_crop(img, (self.params.img_size * super_sampling_factor, self.params.img_size * super_sampling_factor))
        img = F.resize(img, size=(self.params.img_size, self.params.img_size), interpolation=InterpMethod(self.params.affine_interpolation_method).to_torch())
        img = img.squeeze(0)

        # Images for visualization
        img_plot = img.cpu().numpy().astype(np.float32)
        img_plot = normalize_0_to_1(img_plot, based_on_min_max=True)
        if self.img_cmap == 'gray':
            img_plot = np.expand_dims(img_plot, axis=-1)
            img_plot = np.concatenate([img_plot, img_plot, img_plot], axis=-1)
        else:
            img_plot = rutils.apply_color_map(img_plot, self.img_cmap)
            img_plot = cv2.cvtColor(img_plot, cv2.COLOR_BGR2RGB)

        self.state.img = np.ascontiguousarray(img_plot)

        # ---------------------------------------------------------------------------------------------------
        # Compute the FFT

        if self.params.apply_windowing:
            # Apply the Kaiser window
            window = torch.kaiser_window(img.shape[-1], periodic=False, beta=self.params.beta, dtype=img.dtype, device=img.device)
            window *= window.square().sum().rsqrt()

            window_2d = torch.ger(window, window)  # [short_side, short_side]

            img = img * window_2d

            self.state.window = np.ascontiguousarray(window.cpu().numpy().astype(np.float64))

        if self.params.apply_padding:
            # Apply zero padding

            padding = (self.params.img_size * self.params.padding_factor - self.params.img_size)

            img = torch.nn.functional.pad(img, (0, padding, 0, padding))

        # Compute power spectrum density
        spectrum: torch.Tensor = torch.fft.fftn(img, dim=(-2, -1)).abs().square()
        spectrum = torch.fft.fftshift(spectrum, dim=(-2, -1))
        psd = spectrum / (img.shape[-2] * img.shape[-1])

        psd_db = 10.0 * torch.log10(psd + 1e-10)  # to decibels

        self.state.psd_img = np.ascontiguousarray(psd_db.detach().cpu().numpy()).astype(np.float32)

        # ---------------------------------------------------------------------------------------------------
        # Compute the radial power spectrum density
        rad_psd = _module.calc_radial_psd_profile(
            psd.unsqueeze(0).unsqueeze(-1).contiguous(),  # [1, H, W, 1]
            int(360 * 2),
            int(1024)
        )  # [1, n_divs, n_points, 1]

        rad_psd = rad_psd.mean(dim=(0, 1, 3))  # [n_points,]

        self.state.rad_psd = rad_psd.cpu().numpy().astype(np.float64)

        # ---------------------------------------------------------------------------------------------------
        # Copy parameters
        self.state.prev_params = copy.deepcopy(self.params)

        return self.state.img

    @docking_viewer.dockable
    def toolbar(self) -> None:
        """Build the toolbar UI for the Color of Noise visualizer.
        """

        # ---------------------------------------------------------------------------------------------------
        # Plot the power spectrum density

        cmap = getattr(implot.Colormap_, self.psd_cmap, None)
        if cmap is not None:
            implot.push_colormap(cmap.value)

        heatmap_size = 128 * 5
        if self.state.psd_img is not None and implot.begin_plot(
            'Power Spectrum Density',
            size=(heatmap_size, heatmap_size),
            flags=implot.Flags_.no_legend.value
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
                flags=0
            )

            imgui.same_line()
            implot.colormap_scale("Power Spectrum Density [dB]", scale_min, scale_max, size=(20, heatmap_size))

            implot.end_plot()

        if cmap is not None:
            implot.pop_colormap()

        # ---------------------------------------------------------------------------------------------------
        # Plot the radial power spectrum density

        if self.state.rad_psd is not None and imgui.same_line() is None and implot.begin_plot(
            'Radial Power Spectrum Density',
            size=(-1, 512),
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

        # ---------------------------------------------------------------------------------------------------
        # Parameters for the noise
        if imgui.collapsing_header('Input', flags=imgui.TreeNodeFlags_.default_open):
            self.params.img_mode = imgui.combo(
                'Image mode',
                self.params.img_mode,
                [m.name for m in ImageMode],
            )[1]

            self.params.img_path = pathlib.Path(imgui.input_text('Image path', str(self.params.img_path))[1])

            self.params.img_size = imgui.slider_int('Image size', self.params.img_size, 16, 1024)[1]

            imgui.text('Wave number')
            imgui.same_line()
            self.params.wave_number = imgui.slider_float('##Wave number slider', self.params.wave_number, 0.0, float(self.params.img_size))[1]
            imgui.same_line()
            self.params.wave_number = imgui.input_float('##Wave number input', self.params.wave_number)[1]

            self.params.rotate = imgui.slider_float('Rotation angle', self.params.rotate, 0.0, 360.0)[1]

            self.params.enable_super_sampling = imgui.checkbox('Enable super sampling', self.params.enable_super_sampling)[1]
            self.params.super_sampling_factor = imgui.slider_int('Super sampling factor', self.params.super_sampling_factor, 1, 4)[1]

            self.params.enable_pre_filering = imgui.checkbox('Enable pre-filtering', self.params.enable_pre_filering)[1]
            kernel_size = imgui.slider_int('Kernel size', self.params.kernel_size, 3, 31)[1]
            self.params.kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
            self.params.kernel_sigma = imgui.slider_float('Kernel sigma', self.params.kernel_sigma, 0.01, 10.0)[1]

            self.params.affine_interpolation_method = imgui.combo(
                'Affine interp',
                self.params.affine_interpolation_method,
                [m.name for m in InterpMethod],
            )[1]

        # ---------------------------------------------------------------------------------------------------
        # FFT parameters
        if imgui.collapsing_header('FFT', flags=imgui.TreeNodeFlags_.default_open):
            imgui.separator_text('Windowing')
            imgui.push_id('enable_windowing')
            self.params.apply_windowing = imgui.checkbox('Enable', self.params.apply_windowing)[1]
            imgui.pop_id()
            self.params.beta = imgui.slider_float('Beta', self.params.beta, 0.0, 20.0)[1]

            if self.state.window is not None and self.params.apply_windowing and implot.begin_plot('Kaiser Window', size=(-1, 256)):
                implot.setup_axes('Pixel', 'Weight')
                implot.setup_axes_limits(
                    0.0,
                    self.state.window.shape[0],
                    0.0,
                    1.0,
                    imgui.Cond_.always.value,
                )
                implot.plot_line('Kaiser window', self.state.window)
                implot.end_plot()

            imgui.separator_text('Padding')
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
    print('Bye!')
