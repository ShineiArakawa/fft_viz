# autopep8: off
import copy
import enum
import math
import typing

import matplotlib as mpl

# Set Agg backend
mpl.use('Agg')

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydantic.dataclasses as dataclasses
import pyviewer.docking_viewer as docking_viewer
import research_utilities as rutils
import torch
from imgui_bundle import imgui, implot

# autopep8: on

# --------------------------------------------------------------------------------------------------------------------------------------------------------
# Parameters


class InterpMethod(enum.IntEnum):
    # autopep8: off
    nearest                  = 0
    bilinear                 = 1
    licubic                  = 2
    lanczos                  = 3
    # autopep8: on

    def to_cv2(self):
        if self == InterpMethod.nearest:
            return cv2.INTER_NEAREST
        elif self == InterpMethod.bilinear:
            return cv2.INTER_LINEAR
        elif self == InterpMethod.licubic:
            return cv2.INTER_CUBIC
        elif self == InterpMethod.lanczos:
            return cv2.INTER_LANCZOS4


MPL_CMAPS: typing.Final[list[str]] = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'gray']


@dataclasses.dataclass
class Params:
    """Parameters for the Color of Noise visualizer.
    """

    # autopep8: off
    img_size                            : int          = 512                        # image size
    wave_number                         : float        = 1.0                        # wave number
    rotate                              : float        = 0.0                        # rotation angle in degrees
    affine_interpolation_method         : int          = int(InterpMethod.nearest)  # interpolation method for the affine transformation

    apply_windowing                     : bool         = False                      # apply windowing to the image
    beta                                : float        = 8.0                        # beta parameter for the kaiser window
    apply_padding                       : bool         = False                      # apply zero padding to the image
    padding_factor                      : int          = 4                          # padding factor for the FFT

    img_cmap_id                         : int          = 5                          # color map ID for the sinusoidal image, by default 'gray'
    psd_cmap_id                         : int          = 0                          # color map ID for the power spectrum density, by default 'viridis'
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
        super().__init__(name, with_font_awesome=True, with_implot=True)

    def setup_state(self) -> None:
        """Initialize the state of the visualizer. Called by the super class.
        """

        self.state.params = Params()
        self.state.prev_params = None

        self.state.img = None

        self.state.window = None

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

        canvas_size = math.floor(self.params.img_size * 1.5)  # Allocate larger canvas for the rotation
        t = np.arange(canvas_size, dtype=np.float64)
        t = t - canvas_size // 2  # center the image

        x = np.sin(2.0 * np.pi * target_freq * t)
        img = np.tile(x, (len(x), 1))

        # Rotate the image
        rot_mat = cv2.getRotationMatrix2D((canvas_size // 2, canvas_size // 2), self.params.rotate, 1.0)
        img = cv2.warpAffine(img, rot_mat, (canvas_size, canvas_size), flags=InterpMethod(self.params.affine_interpolation_method).to_cv2())

        # Center crop the image
        x1 = int(img.shape[0] / 2 - self.params.img_size / 2)
        x2 = int(img.shape[0] / 2 + self.params.img_size / 2)
        y1 = int(img.shape[1] / 2 - self.params.img_size / 2)
        y2 = int(img.shape[1] / 2 + self.params.img_size / 2)
        img = img[y1:y2, x1:x2]

        # Images for visualization
        img_plot = img.astype(np.float32)
        img_plot = normalize_0_to_1(img_plot, based_on_min_max=True)
        if self.img_cmap == 'gray':
            img_plot = np.expand_dims(img_plot, axis=-1)
            img_plot = np.concatenate([img_plot, img_plot, img_plot], axis=-1)
        else:
            img_plot = rutils.apply_color_map(img_plot, self.img_cmap)
            img_plot = cv2.cvtColor(img_plot, cv2.COLOR_BGR2RGB)

        # ---------------------------------------------------------------------------------------------------
        # Compute the FFT

        img = to_cuda_device(torch.from_numpy(img)).to(torch.float64)

        if self.params.apply_windowing:
            # Apply the Kaiser window
            window = torch.kaiser_window(img.shape[-1], periodic=False, beta=self.params.beta, dtype=torch.float64, device=img.device)
            window *= window.square().sum().rsqrt()

            window_2d = torch.ger(window, window)  # [short_side, short_side]

            img = img * window_2d

            self.state.window = np.ascontiguousarray(window.cpu().numpy().astype(np.float64))

        if self.params.apply_padding:
            # Apply zero padding

            padding = (self.params.img_size * self.params.padding_factor - self.params.img_size)

            img = torch.nn.functional.pad(img, (0, padding, 0, padding))

        # Compute power spectrum density
        spectrum = torch.fft.fftn(img, dim=(-2, -1)).abs().square()
        spectrum = torch.fft.fftshift(spectrum, dim=(-2, -1))
        psd = spectrum / (img.shape[-2] * img.shape[-1])

        psd = 10.0 * torch.log10(psd + 1e-10)  # to decibels

        psd_img = psd.cpu().numpy()

        # Plot
        fig = plt.figure(figsize=(8, 8), dpi=400, layout='constrained')
        axes = fig.add_subplot(1, 1, 1)
        axes.imshow(psd_img, cmap=self.psd_cmap, extent=(-psd_img.shape[1] // 2, psd_img.shape[1] // 2, -psd_img.shape[0] // 2, psd_img.shape[0] // 2))
        axes.set_title('Power Spectrum Density')
        axes.set_xlabel('Frequency [wave number]')
        axes.set_ylabel('Frequency [wave number]')
        fig.colorbar(axes.images[0], ax=axes, orientation='vertical', pad=0.02, shrink=0.8, label='Power Spectrum Density [dB]')
        fig.canvas.draw()
        psd_img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).copy()
        fig_size = fig.canvas.get_width_height()
        plt.close(fig)
        plt.clf()
        del fig

        psd_img = np.reshape(psd_img, (fig_size[1], fig_size[0], 4))
        psd_img = cv2.cvtColor(psd_img, cv2.COLOR_RGBA2RGB).astype(np.float32)
        psd_img = psd_img / 255.0

        # ---------------------------------------------------------------------------------------------------
        # concatenate the images

        assert psd_img.shape[0] >= img_plot.shape[0], f"psd_img.shape[0]: {psd_img.shape[0]} >= img_plot.shape[0]: {img_plot.shape[0]}"

        required_height = psd_img.shape[0]
        required_width = int(2 * psd_img.shape[1] * 1.05)

        all_img = np.zeros((required_height, required_width, 3), dtype=np.float32)

        img_width = psd_img.shape[1]
        all_img[:, :img_width, :] = cv2.resize(img_plot, (img_width, required_height), interpolation=cv2.INTER_NEAREST)

        all_img[:, required_width - img_width:, :] = cv2.resize(psd_img, (img_width, required_height), interpolation=cv2.INTER_NEAREST)

        # ---------------------------------------------------------------------------------------------------
        # Copy parameters
        self.state.prev_params = copy.deepcopy(self.params)
        self.state.img = all_img

        return self.state.img

    @docking_viewer.dockable
    def toolbar(self) -> None:
        """Build the toolbar UI for the Color of Noise visualizer.
        """

        # ---------------------------------------------------------------------------------------------------
        # Parameters for the noise
        if imgui.collapsing_header('Input', flags=imgui.TreeNodeFlags_.default_open):
            self.params.img_size = imgui.slider_int('Image size', self.params.img_size, 16, 1024)[1]

            imgui.text('Wave number')
            imgui.same_line()
            self.params.wave_number = imgui.slider_float('##Wave number slider', self.params.wave_number, 0.0, float(self.params.img_size))[1]
            imgui.same_line()
            self.params.wave_number = imgui.input_float('##Wave number input', self.params.wave_number)[1]

            self.params.rotate = imgui.slider_float('Rotation angle', self.params.rotate, 0.0, 360.0)[1]

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
