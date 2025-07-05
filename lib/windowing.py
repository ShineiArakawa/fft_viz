"""Collection of windowing functions for signal processing.
See also: https://en.wikipedia.org/wiki/Window_function
"""

# isort: skip_file
# autopep8: off
import abc
import copy
import math
import typing

import torch

import lib.util as util
from imgui_bundle import imgui
# autopep8: on

logger = util.get_logger()


# --------------------------------------------------------------------------------------------------------------------------------------------------------
# Windowing functions


class WindowFunctionBase(metaclass=abc.ABCMeta):
    """Base class for windowing functions.
    """

    def __init__(self):
        self._params: dict[str, util.ValueEntity] = {}

    @property
    def params(self) -> dict[str, util.ValueEntity]:
        pass

    @params.setter
    def params(self, params: dict[str, util.ValueEntity]) -> None:
        self._params = copy.deepcopy(params)

    @params.getter  # See also: https://en.wikipedia.org/wiki/Window_function
    def params(self) -> dict[str, util.ValueEntity]:
        return self._params

    @abc.abstractmethod
    def calc_window(self, size: int, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        pass


class RectangualrWindow(WindowFunctionBase):
    def __init__(self):
        super().__init__()

    def calc_window(self, size: int, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        window = torch.ones((size,), dtype=dtype, device=device)
        window *= window.square().sum().rsqrt()
        window_2d = torch.ger(window, window)  # [win_size, win_size]
        return window, window_2d


class TriangularWindow(WindowFunctionBase):
    def __init__(self):
        super().__init__()

    def calc_window(self, size: int, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        half_size = (size + 2 - 1) // 2
        window = torch.linspace(0.0, 1.0, half_size, dtype=dtype, device=device)
        start_idx = 0 if size % 2 == 0 else 1
        window = torch.cat([window, window.flip(0)[start_idx:]], dim=0)
        assert window.shape[0] == size, f'Window size mismatch: {window.shape[0]} != {size}'
        window *= window.square().sum().rsqrt()
        window_2d = torch.ger(window, window)  # [win_size, win_size]
        return window, window_2d


class ParzenWindow(WindowFunctionBase):
    def __init__(self):
        super().__init__()

    def calc_window(self, size: int, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        N = size - 1
        L = N + 1

        n = torch.arange(0, N + 1, device=device).to(dtype=dtype)  # [0, 1, ..., N]
        n = n - N / 2  # Center the window around 0

        window = torch.zeros((size,), dtype=dtype, device=device)

        skirt_idx = (n.abs() <= (L / 4))
        window[skirt_idx] = 1 - 6.0 * ((2.0 * n[skirt_idx] / L) ** 2) * (1.0 - 2.0 * n[skirt_idx].abs() / L)

        center_idx = ~skirt_idx
        window[center_idx] = 2.0 * (1.0 - 2.0 * n[center_idx].abs() / L) ** 3

        window *= window.square().sum().rsqrt()
        window_2d = torch.ger(window, window)  # [win_size, win_size]

        return window, window_2d


class WelchWindow(WindowFunctionBase):
    def __init__(self):
        super().__init__()

    def calc_window(self, size: int, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        N = size - 1

        n = torch.arange(0, N + 1, device=device).to(dtype=dtype)  # [0, 1, ..., N]

        window = 1.0 - ((n - N / 2.0) / (N / 2.0)) ** 2

        # window = torch.clamp(window, min=0.0, max=1.0)
        window *= window.square().sum().rsqrt()

        window_2d = torch.ger(window, window)  # [win_size, win_size]
        return window, window_2d


class HannWindow(WindowFunctionBase):
    def __init__(self):
        super().__init__()

    def calc_window(self, size: int, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        window = torch.hann_window(window_length=size, dtype=dtype, device=device, periodic=False)
        window *= window.square().sum().rsqrt()
        window_2d = torch.ger(window, window)  # [win_size, win_size]
        return window, window_2d


class HammingWindow(WindowFunctionBase):
    # See also: https://docs.pytorch.org/docs/stable/generated/torch.hamming_window.html

    def __init__(self):
        super().__init__()

        self._params = {
            'alpha': util.ValueEntity(value=25.0 / 46.0, value_type=float, min_value=0.0, max_value=1.0),
            'beta': util.ValueEntity(value=1.0 - 0.46164, value_type=float, min_value=0.0, max_value=1.0),
        }

    def calc_window(self, size: int, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = float(self._params['alpha'].value)
        beta = float(self._params['beta'].value)

        window = torch.hamming_window(window_length=size, dtype=dtype, device=device, alpha=alpha, beta=beta, periodic=False)
        window *= window.square().sum().rsqrt()
        window_2d = torch.ger(window, window)  # [win_size, win_size]
        return window, window_2d


class ExponentialWindow(WindowFunctionBase):
    def __init__(self):
        super().__init__()

        self._params = {
            'tau': util.ValueEntity(value=1.0, value_type=float, min_value=0.01, max_value=10.0),
        }

    def calc_window(self, size: int, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        tau = float(self._params['tau'].value)
        window = torch.windows.exponential(size, dtype=dtype, device=device, tau=tau)
        window *= window.square().sum().rsqrt()
        window_2d = torch.ger(window, window)  # [win_size, win_size]
        return window, window_2d


class ExponentialSymmetricWindow(WindowFunctionBase):
    def __init__(self):
        super().__init__()

        self._params = {
            'sigma': util.ValueEntity(value=1.0, value_type=float, min_value=0.01, max_value=5.0),
        }

    def calc_window(self, size: int, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        sigma = float(self._params['sigma'].value)
        t = torch.linspace(-1.0, 1.0, size, dtype=dtype, device=device)
        x, y = torch.meshgrid(t, t, indexing='ij')
        r = (x ** 2 + y ** 2).sqrt()

        window = torch.exp(- t.abs() / sigma)
        window *= window.square().sum().rsqrt()

        window_2d = torch.exp(- r.abs() / sigma)
        window_2d *= window_2d.square().sum().rsqrt()

        return window, window_2d


class NuttallWindow(WindowFunctionBase):
    def __init__(self):
        super().__init__()

    def calc_window(self, size: int, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        window = torch.windows.nuttall(size, dtype=dtype, device=device)
        window *= window.square().sum().rsqrt()
        window_2d = torch.ger(window, window)  # [win_size, win_size]
        return window, window_2d


class BackmanWindow(WindowFunctionBase):
    def __init__(self):
        super().__init__()

    def calc_window(self, size: int, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        window = torch.blackman_window(size, dtype=dtype, device=device)
        window *= window.square().sum().rsqrt()
        window_2d = torch.ger(window, window)  # [win_size, win_size]
        return window, window_2d


class FlatTopWindow(WindowFunctionBase):
    def __init__(self):
        super().__init__()

        self._params = {
            'a_0': util.ValueEntity(value=0.21557895, value_type=float, min_value=0.0, max_value=1.0),
            'a_1': util.ValueEntity(value=0.41663158, value_type=float, min_value=0.0, max_value=1.0),
            'a_2': util.ValueEntity(value=0.277263158, value_type=float, min_value=0.0, max_value=1.0),
            'a_3': util.ValueEntity(value=0.083578947, value_type=float, min_value=0.0, max_value=1.0),
            'a_4': util.ValueEntity(value=0.006947368, value_type=float, min_value=0.0, max_value=1.0),
        }

    def calc_window(self, size: int, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        N = size - 1
        n = torch.arange(0, N + 1, device=device).to(dtype=dtype)  # [0, 1, ..., N]

        a_0 = float(self._params['a_0'].value)
        a_1 = float(self._params['a_1'].value)
        a_2 = float(self._params['a_2'].value)
        a_3 = float(self._params['a_3'].value)
        a_4 = float(self._params['a_4'].value)

        window = a_0 - \
            a_1 * torch.cos(2 * math.pi * n / N) + \
            a_2 * torch.cos(4 * math.pi * n / N) - \
            a_3 * torch.cos(6 * math.pi * n / N) + \
            a_4 * torch.cos(8 * math.pi * n / N)

        window *= window.square().sum().rsqrt()

        window_2d = torch.ger(window, window)  # [win_size, win_size]
        return window, window_2d


class KaiserWindow(WindowFunctionBase):
    def __init__(self):
        super().__init__()

        self._params = {
            'beta': util.ValueEntity(value=8.0, value_type=float, min_value=0.0, max_value=100.0),
        }

    def calc_window(self, size: int, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        beta = float(self._params['beta'].value)
        window = torch.kaiser_window(size, periodic=False, dtype=dtype, device=device, beta=beta)
        window *= window.square().sum().rsqrt()
        window_2d = torch.ger(window, window)  # [win_size, win_size]
        return window, window_2d


class TukeyWindow(WindowFunctionBase):
    def __init__(self):
        super().__init__()

        self._params = {
            'alpha': util.ValueEntity(value=0.5, value_type=float, min_value=0.0, max_value=1.0),
        }

    def calc_window(self, size: int, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = float(self._params['alpha'].value)
        N = size - 1

        bounb_0 = math.ceil(alpha * N / 2.0)
        bounb_1 = N // 2

        window = torch.arange(size, dtype=dtype, device=device)
        window[:bounb_0] = (1.0 - torch.cos(2.0 * torch.pi * window[:bounb_0] / (alpha * N))) / 2.0
        window[bounb_0:bounb_1 + 1] = 1.0
        window[-(bounb_1 + 1):] = window[:bounb_1 + 1].flip(0)

        window *= window.square().sum().rsqrt()
        window_2d = torch.ger(window, window)  # [win_size, win_size]

        return window, window_2d


class LanczosWindow(WindowFunctionBase):
    def __init__(self):
        super().__init__()

    def calc_window(self, size: int, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        N = size - 1
        n = torch.arange(0, N + 1, device=device).to(dtype=dtype)  # [0, 1, ..., N]
        window = torch.sinc(2.0 * n / N - 1.0)

        window *= window.square().sum().rsqrt()
        window_2d = torch.ger(window, window)  # [win_size, win_size]

        return window, window_2d  # Lanczos window is normalized by its sum, not square root of sum of squares.


_window_func_names: typing.Final[tuple[str]] = (
    'rectangular',
    'triangular',
    'Parzen',
    # ----------
    # Polynomial windows
    'Welch',
    'Hann',
    'Hamming',
    # ----------
    # Exponential windows
    'exponential',
    'exponential_sym',
    # ----------
    # Raised cosine windows
    'Nuttall',
    'Blackman',
    'flattop',
    # ----------
    # Other windows
    'Tukey (cosine-tapered)',
    'Kaiser',
    'Lanczos',
)

_window_funcs: dict[str, typing.Type[WindowFunctionBase]] = {
    # autopep8: off
    'rectangular'                     : RectangualrWindow,
    'triangular'                      : TriangularWindow,
    'Parzen'                          : ParzenWindow,
    'Welch'                           : WelchWindow,
    'Hann'                            : HannWindow,
    'Hamming'                         : HammingWindow,
    'exponential'                     : ExponentialWindow,
    'exponential_sym'                 : ExponentialSymmetricWindow,
    'Nuttall'                         : NuttallWindow,
    'Blackman'                        : BackmanWindow,
    'flattop'                         : FlatTopWindow,
    'Kaiser'                          : KaiserWindow,
    'Tukey (cosine-tapered)'          : TukeyWindow,
    'Lanczos'                         : LanczosWindow,
    # autopep8: on
}

# -------------------------------------------------------------------------------------------------------------------------------------------------
# Visualizer for debugging

if __name__ == '__main__':
    import numpy as np
    import pyviewer_extended
    from imgui_bundle import implot

    dtype = torch.float64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class WindowingDemo(pyviewer_extended.MultiTexturesDockingViewer):
        # autopep8: off
        KEY_WINDOW        : typing.Final[str] = 'Window'
        # autopep8: on

        KEYS = [KEY_WINDOW,]

        def __init__(self, name):
            # ------------------------------------------------------------------------------------
            super().__init__(name, self.KEYS, with_font_awesome=True, with_implot=True)

        def setup_state(self):
            # Instantiate all the window functions
            self.state.window_func = {}
            for name, cls in _window_funcs.items():
                self.state.window_func[name] = cls()

            self.state.window_size = 64
            self.state.window = 0

            self.state.window_1d = None
            self.state.window_img = None

            self.state.psd_img = None

        def compute(self):
            window_func: WindowFunctionBase = self.state.window_func[_window_func_names[self.state.window]]

            window_1d, window_2d = window_func.calc_window(
                self.state.window_size, dtype=dtype, device=device
            )

            self.state.window_img = util.normalize_0_to_1(window_2d.unsqueeze(-1).tile(1, 1, 3).to(torch.float32),  based_on_min_max=True)
            self.state.window_1d = np.ascontiguousarray(window_1d.cpu().numpy().astype(np.float64))

            # fft
            spectrum = torch.fft.fftn(window_2d, dim=(-2, -1)).abs().square()
            spectrum = torch.fft.fftshift(spectrum, dim=(-2, -1))
            psd = spectrum / (window_2d.shape[-2] * window_2d.shape[-1])
            psd_plot = 10.0 * torch.log10(psd + 1e-10)  # to decibels
            self.state.psd_img = np.ascontiguousarray(psd_plot.cpu().numpy()).astype(np.float32)

            return {
                self.KEY_WINDOW: self.state.window_img,
            }

        @pyviewer_extended.dockable
        def window_plot(self):
            window_name = _window_func_names[self.state.window]
            if self.state.window_1d is not None and implot.begin_plot(f'{window_name} Window', size=(-1, -1)):
                implot.setup_axes('Pixel', 'Weight')
                implot.setup_axes_limits(
                    0.0,
                    self.state.window_1d.shape[0] - 1,
                    -0.3,
                    1.3,
                    imgui.Cond_.always.value,
                )
                implot.plot_line(window_name, self.state.window_1d)
                implot.end_plot()

        @pyviewer_extended.dockable
        def window_psd_plot(self):
            x_avail, _ = imgui.get_content_region_avail()
            color_bar_prop = 0.10  # 10% for the color bar
            plot_width = x_avail * (1.0 - color_bar_prop)
            color_bar_width = x_avail - plot_width

            cmap = getattr(implot.Colormap_, 'plasma', None)
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

                    half_size = self.state.window_size // 2

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

        @pyviewer_extended.dockable
        def toolbar(self):
            self.state.window_size = imgui.slider_int(
                'Window Size', self.state.window_size, 8, 1024
            )[1]
            self.state.window = imgui.combo(
                'Window Function', self.state.window, _window_func_names
            )[1]
            imgui.separator_text('Parameters')
            window_func: WindowFunctionBase = self.state.window_func[_window_func_names[self.state.window]]
            for name, param in window_func.params.items():
                param.draw_param_widgets(name)
            pass

    _ = WindowingDemo('Windowing Functions Demo')

# -------------------------------------------------------------------------------------------------------------------------------------------------
