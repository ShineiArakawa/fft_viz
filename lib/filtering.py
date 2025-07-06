"""Collection of filter functions.
"""

from __future__ import annotations

import abc
import copy
import math
import typing

import torch

import lib.util as util

# --------------------------------------------------------------------------------------------------------------------------------------------------------
# Base class


class FilterBase(metaclass=abc.ABCMeta):
    """Base class for filters.
    """

    def __init__(self):
        self._params: dict[str, util.ValueEntity] = {}

    @property
    def params(self) -> dict[str, util.ValueEntity]:
        pass

    @params.setter
    def params(self, params: dict[str, util.ValueEntity]) -> None:
        self._params = copy.deepcopy(params)

    @params.getter
    def params(self) -> dict[str, util.ValueEntity]:
        return self._params

    @abc.abstractmethod
    def compute_filter(self, size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        pass

    def draw_param_widgets(self) -> None:
        for param_name, param_value in self.params.items():
            param_value.draw_param_widgets(param_name)

# --------------------------------------------------------------------------------------------------------------------------------------------------------
# On-op Filter


class AllPassFilter(FilterBase):
    """no-op filter
    """

    def compute_filter(self, size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        filter = torch.ones((size, size), dtype=dtype, device=device)  # [size, size]
        return filter

# --------------------------------------------------------------------------------------------------------------------------------------------------------
# Ideal Filter


class IdealLowPassFilter(FilterBase):
    """Low-pass filter with a sharp cutoff in frequency domain.
    This filter is not practical because such sharp cut-off causes ringing artifact in spatial domain.

    See also:
      - https://en.wikipedia.org/wiki/Sinc_filter
      - https://en.wikipedia.org/wiki/Ringing_artifacts
    """

    def __init__(self):
        super().__init__()

        self._params = {'Cut-off freq': util.ValueEntity(value=10.0, value_type=float, min_value=0.0, max_value=1024)}

    def compute_filter(self, size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        f_cut = float(self._params['Cut-off freq'].value)

        freq = torch.fft.fftfreq(n=size, d=1.0 / size).to(dtype=dtype, device=device)  # [size,]
        freq = torch.fft.fftshift(freq)
        freq_x, freq_y = torch.meshgrid((freq, freq), indexing='ij')  # [size, size]

        filter = torch.zeros_like(freq_x)  # [size, size]
        passband_idx = (freq_x ** 2 + freq_y ** 2) < f_cut * f_cut
        filter[passband_idx] = 1.0

        return filter


class IdealHighPassFilter(FilterBase):
    """High-pass filter with a sharp cutoff in frequency domain.
    This filter is not practical because such sharp cut-off causes ringing artifact in spatial domain.

    See also:
      - https://en.wikipedia.org/wiki/Sinc_filter
      - https://en.wikipedia.org/wiki/Ringing_artifacts
    """

    def __init__(self):
        super().__init__()

        self._params = {'Cut-off freq': util.ValueEntity(value=10.0, value_type=float, min_value=0.0, max_value=1024)}

    def compute_filter(self, size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        f_cut = float(self._params['Cut-off freq'].value)

        freq = torch.fft.fftfreq(n=size, d=1.0 / size).to(dtype=dtype, device=device)  # [size,]
        freq = torch.fft.fftshift(freq)
        freq_x, freq_y = torch.meshgrid((freq, freq), indexing='ij')  # [size, size]

        filter = torch.zeros_like(freq_x)  # [size, size]
        passband_idx = (freq_x ** 2 + freq_y ** 2) > f_cut * f_cut
        filter[passband_idx] = 1.0

        return filter

# --------------------------------------------------------------------------------------------------------------------------------------------------------
# Butterworth Filter


class ButterworthFilterBase(FilterBase):
    def __init__(self):
        super().__init__()

        self._params = {
            'Cut-off freq': util.ValueEntity(value=10.0, value_type=float, min_value=0.0, max_value=1024),
            'order': util.ValueEntity(value=5, value_type=int, min_value=1, max_value=100),
        }

    def _compute_w(self, size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        f_cut = float(self._params['Cut-off freq'].value)
        n = int(self._params['order'].value)  # [wave number]

        freq = torch.fft.fftfreq(n=size, d=1.0 / size).to(dtype=dtype, device=device)  # [wave number]
        freq = torch.fft.fftshift(freq)

        freq = freq / f_cut  # Scale to have freq == 1.0 at cut-off freq point

        freq_x, freq_y = torch.meshgrid((freq, freq), indexing='ij')  # [size, size]
        w_squared = freq_x ** 2 + freq_y ** 2
        w_squared = w_squared.pow(n)

        return w_squared


class ButterworthLowPassFilter(ButterworthFilterBase):
    """'Butterworth' low-pass filter, which has a almost flat response in passband.

    See also:
      - https://en.wikipedia.org/wiki/Butterworth_filter
      - https://scikit-image.org/docs/stable/auto_examples/filters/plot_butterworth.html
      - https://github.com/scikit-image/scikit-image/blob/e8a42ba85aaf5fd9322ef9ca51bc21063b22fcae/skimage/filters/_fft_based.py#L9
    """

    def compute_filter(self, size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        w = self._compute_w(size, dtype, device)
        filter = (1.0 + w).rsqrt()
        return filter


class ButterworthHighPassFilter(ButterworthFilterBase):
    """'Butterworth' high-pass filter, which has a almost flat response in passband.

    See also:
      - https://en.wikipedia.org/wiki/Butterworth_filter
      - https://scikit-image.org/docs/stable/auto_examples/filters/plot_butterworth.html
      - https://github.com/scikit-image/scikit-image/blob/e8a42ba85aaf5fd9322ef9ca51bc21063b22fcae/skimage/filters/_fft_based.py#L9
    """

    def compute_filter(self, size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        w = self._compute_w(size, dtype, device)
        filter = (w / (1.0 + w)).sqrt()
        return filter

# --------------------------------------------------------------------------------------------------------------------------------------------------------
# Chebyshev Filter


@torch.jit.script
def chebyshev_poly1(x: torch.Tensor, n: int) -> torch.Tensor:
    """Compute type I chebyshev polynomials with recurrence computation

    See also:
        - https://en.wikipedia.org/wiki/Chebyshev_polynomials
    """

    if n == 0:
        return torch.ones_like(x)
    elif n == 1:
        return x

    t_0 = torch.ones_like(x)
    t_1 = x

    for _ in range(1, n):
        t_2 = 2.0 * x * t_1 - t_0
        t_0, t_1 = t_1, t_2

    return t_1


class ChebyshevFilterBase(FilterBase):
    def __init__(self):
        super().__init__()

        self._params = {
            'Cut-off freq': util.ValueEntity(value=10.0, value_type=float, min_value=0.0, max_value=1024),
            'Ripple gain': util.ValueEntity(value=-3.0, value_type=float, min_value=-10.0, max_value=0.0),
            'Chebyshev poly order': util.ValueEntity(value=4, value_type=int, min_value=0, max_value=10),
        }

    @abc.abstractmethod
    def compute_eps(self, delta: float):
        pass

    def _compute_w(self, size: int, dtype: torch.dtype, device: torch.device, flip_freq: bool = False) -> torch.Tensor:
        f_cut = float(self._params['Cut-off freq'].value)
        delta = float(self._params['Ripple gain'].value)
        n = int(self._params['Chebyshev poly order'].value)

        eps = self.compute_eps(delta)

        freq = torch.fft.fftfreq(n=size, d=1.0 / size).to(dtype=dtype, device=device)  # [wave number]
        freq = torch.fft.fftshift(freq)

        freq_x, freq_y = torch.meshgrid((freq, freq), indexing='ij')  # [size, size]

        w = (freq_x ** 2 + freq_y ** 2).sqrt()
        w = f_cut / (w + 1e-20) if flip_freq else w / (f_cut + 1e-20)
        w = (eps * chebyshev_poly1(w, n)).square()

        return w


class Chebyshev1LowPassFilter(ChebyshevFilterBase):
    """Type I Chebyshev low-pass filter, which has a sharper roll-off than that of betterworth filter. It also has a oscillating gain, called 'ripple,' within passband.

    See also:
      - https://en.wikipedia.org/wiki/Chebyshev_filter
      - https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cheby1.html
    """

    def compute_eps(self, delta: float):
        eps = math.sqrt(10.0 ** (- delta / 10.0) - 1.0)
        return eps

    def compute_filter(self, size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        w = self._compute_w(size, dtype, device)
        filter = (1.0 + w).rsqrt()
        return filter


class Chebyshev1HighPassFilter(ChebyshevFilterBase):
    """Type I Chebyshev high-pass filter, which has a sharper roll-off than that of betterworth filter. It also has a oscillating gain, called 'ripple,' within passband.

    See also:
      - https://en.wikipedia.org/wiki/Chebyshev_filter
      - https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cheby1.html
    """

    def compute_eps(self, delta: float):
        eps = math.sqrt(10.0 ** (- delta / 10.0) - 1.0)
        return eps

    def compute_filter(self, size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        w = self._compute_w(size, dtype, device, flip_freq=True)
        filter = (1.0 + w).rsqrt()
        return filter


class Chebyshev2LowPassFilter(ChebyshevFilterBase):
    """Type II Chebyshev low-pass filter, which has a sharper roll-off than that of betterworth filter. It also has a oscillating gain, called 'ripple,' within stopband.

    See also:
      - https://en.wikipedia.org/wiki/Chebyshev_filter
      - https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cheby2.html
    """

    def compute_eps(self, delta: float):
        d = 10.0 ** (delta / 10.0)
        return math.sqrt(d / (1.0 - d))

    def compute_filter(self, size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        w = self._compute_w(size, dtype, device, flip_freq=True)
        filter = (1.0 + 1.0 / w).rsqrt()
        return filter


class Chebyshev2HighPassFilter(ChebyshevFilterBase):
    """Type II Chebyshev high-pass filter, which has a sharper roll-off than that of betterworth filter. It also has a oscillating gain, called 'ripple,' within stopband.

    See also:
      - https://en.wikipedia.org/wiki/Chebyshev_filter
      - https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cheby2.html
    """

    def compute_eps(self, delta: float):
        d = 10.0 ** (delta / 10.0)
        return math.sqrt(d / (1.0 - d))

    def compute_filter(self, size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        w = self._compute_w(size, dtype, device)
        filter = (1.0 + 1.0 / w).rsqrt()
        return filter


# --------------------------------------------------------------------------------------------------------------------------------------------------------
# Filter registry
filters: typing.Final[dict[str, typing.Type[FilterBase]]] = {
    # autopep8: off
    'All pass'                           : AllPassFilter,
    'Ideal low-pass'                     : IdealLowPassFilter,
    'Ideal high-pass'                    : IdealHighPassFilter,
    'Butterworth low-pass'               : ButterworthLowPassFilter,
    'Butterworth high-pass'              : ButterworthHighPassFilter,
    'Chebyshev I low-pass '              : Chebyshev1LowPassFilter,
    'Chebyshev I high-pass '             : Chebyshev1HighPassFilter,
    'Chebyshev II low-pass '             : Chebyshev2LowPassFilter,
    'Chebyshev II high-pass '            : Chebyshev2HighPassFilter,
    # autopep8: on
}

filter_class_to_name:  typing.Final[dict[typing.Type[FilterBase], str]] = {value: key for key, value in filters.items()}

filter_names: typing.Final[tuple[str]] = tuple(sorted(filters.keys()))

# -------------------------------------------------------------------------------------------------------------------------------------------------
# Visualizer for debugging

if __name__ == '__main__':
    import numpy as np
    import pyviewer_extended
    from imgui_bundle import hello_imgui, imgui, implot  # type: ignore

    dtype = torch.float64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class FilterDemo(pyviewer_extended.MultiTexturesDockingViewer):
        def __init__(self, name):
            # ------------------------------------------------------------------------------------
            super().__init__(name, [], full_screen_mode=hello_imgui.FullScreenMode.full_monitor_work_area, with_font_awesome=True, with_implot=True)

        def setup_state(self):
            # Instantiate all the filter classes
            self.state.filters = [filters[filter_name]() for filter_name in filter_names]
            self.state.cur_filter_idx = 1
            self.state.size = 1024

            self.state.filter_img = None
            self.state.filter_response_h = None
            self.state.filter_response_v = None

            self.state.response_ylim_fixed = True
            self.state.response_ylim_min = -10
            self.state.response_ylim_max = 0.5
            self.state.response_xscale_log = False

        def compute(self):
            size = self.state.size
            filter_inst: FilterBase = self.state.filters[self.state.cur_filter_idx]
            filter_2d = filter_inst.compute_filter(size, dtype=dtype, device=device)

            filter_img = filter_2d.detach().cpu().numpy().astype(np.float64)
            self.state.filter_img = 20.0 * np.log10(filter_img + 1e-15)

            half_size = size // 2
            self.state.filter_response_h = np.ascontiguousarray(np.fft.ifftshift(filter_img[half_size, :])[:half_size], dtype=np.float64)
            self.state.filter_response_v = np.ascontiguousarray(np.fft.ifftshift(filter_img[:, half_size])[:half_size], dtype=np.float64)

            return None

        @pyviewer_extended.dockable
        def filter_plot(self) -> None:
            # ---------------------------------------------------------------------------------------------------
            # Plot the filter

            filter_cmap = 'viridis'

            x_avail, _ = imgui.get_content_region_avail()
            color_bar_prop = 0.10  # 10% for the color bar
            plot_width = x_avail * (1.0 - color_bar_prop)
            color_bar_width = x_avail - plot_width

            cmap = getattr(implot.Colormap_, filter_cmap, None)
            if cmap is not None:
                implot.push_colormap(cmap.value)

            if self.state.filter_img is not None:
                filter_img = self.state.filter_img
                scale_min = np.min(filter_img)
                scale_max = np.max(filter_img)

                if implot.begin_plot(
                    'Filter',
                    size=(plot_width, -1),
                    flags=implot.Flags_.no_legend.value | implot.Flags_.equal.value
                ):

                    half_size = self.state.size // 2

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
        def filter_profile_plot(self):
            if imgui.begin_tab_bar('Filter Profiles'):
                for label, filter_response in (('Horizontal', self.state.filter_response_h), ('Vertical', self.state.filter_response_v)):
                    if imgui.begin_tab_item_simple(label):
                        if filter_response is not None and implot.begin_plot(
                            f'{label} Filter Response',
                            size=(-1, -1),
                            flags=implot.Flags_.no_legend.value
                        ):
                            filter_response = np.log10(filter_response.copy() ** 20 + 1e-15)  # NOTE: dB for not power but amplitude: 20 * log10(psd)

                            freq = np.fft.fftfreq(self.state.size, d=1.0/self.state.size)[1:len(filter_response)].astype(np.float64)  # assume that the img is square

                            # Setup x axis
                            implot.setup_axis(implot.ImAxis_.x1, "Frequency [wave number]")
                            implot.setup_axis_limits(implot.ImAxis_.x1, freq.min(), freq.max(), imgui.Cond_.always.value)
                            if self.state.response_ylim_fixed:
                                implot.setup_axis_limits(implot.ImAxis_.y1, self.state.response_ylim_min, self.state.response_ylim_max, imgui.Cond_.always.value)

                            # Setup y axis
                            implot.setup_axis(implot.ImAxis_.y1, "Gain [dB]", flags=implot.AxisFlags_.auto_fit.value)

                            # Set log-log scale
                            implot.setup_axis_scale(implot.ImAxis_.x1, implot.Scale_.log10.value if self.state.response_xscale_log else implot.Scale_.linear.value)
                            implot.setup_axis_scale(implot.ImAxis_.y1, implot.Scale_.linear.value)

                            util.make_unique(implot.plot_line, f'{label} Response', freq, np.ascontiguousarray(filter_response[1:]))

                            implot.end_plot()
                        imgui.end_tab_item()
                imgui.end_tab_bar()

        @pyviewer_extended.dockable
        def toolbar(self):
            imgui.separator_text('Common')
            self.state.size = max(imgui.slider_int('Size', self.state.size, 5, 4096)[1], 5)

            imgui.separator_text('Filter')
            self.state.cur_filter_idx = imgui.combo('Filter', self.state.cur_filter_idx, filter_names)[1]

            cur_filter: FilterBase = self.state.filters[self.state.cur_filter_idx]
            cur_filter.draw_param_widgets()

            imgui.separator_text('Visualization')
            # autopep8: off
            imgui.push_id('Filter Response (Fix y-limits)'); self.state.response_ylim_fixed = imgui.checkbox('Fix y-limits', self.state.response_ylim_fixed)[1]; imgui.pop_id()
            if self.state.response_ylim_fixed:
                imgui.push_id('Filter Response (Min y-limit)'); self.state.response_ylim_min = imgui.input_float('Min y-limit', self.state.response_ylim_min, step=1e-10, format='%.2e')[1]; imgui.pop_id()
                imgui.push_id('Filter Response (Max y-limit)'); self.state.response_ylim_max = imgui.input_float('Max y-limit', self.state.response_ylim_max, step=1e-10, format='%.2e')[1]; imgui.pop_id()
            imgui.push_id('Filter Response (X in log scale)'); self.state.response_xscale_log = imgui.checkbox('X in log scale', self.state.response_xscale_log)[1]; imgui.pop_id()
            # autopep8: on

    _ = FilterDemo('FIR Filter Demo')

# -------------------------------------------------------------------------------------------------------------------------------------------------
