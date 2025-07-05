"""Collection of filter functions.
"""

from __future__ import annotations

import abc
import copy
import typing

import torch

import lib.util as util

# --------------------------------------------------------------------------------------------------------------------------------------------------------
# Base class for easy parameter management by imgui


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

# --------------------------------------------------------------------------------------------------------------------------------------------------------
# Filters


class AllPassFilter(FilterBase):  # no-op
    def compute_filter(self, size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        filter = torch.ones((size, size), dtype=dtype, device=device)  # [size, size]
        return filter


class IdealLowPassFilter(FilterBase):
    def __init__(self):
        super().__init__()

        self._params = {
            'Cut-off freq': util.ValueEntity(value=10.0, value_type=float, min_value=0.0, max_value=1024),
        }

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
    def __init__(self):
        super().__init__()

        self._params = {
            'Cut-off freq': util.ValueEntity(value=10.0, value_type=float, min_value=0.0, max_value=1024),
        }

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
# Registry


filters: typing.Final[dict[str, typing.Type[FilterBase]]] = {
    # autopep8: off
    'All pass'                           : AllPassFilter,
    'Ideal low pass'                     : IdealLowPassFilter,
    'Ideal high pass'                    : IdealHighPassFilter,
    # autopep8: on
}

filter_class_to_name:  typing.Final[dict[typing.Type[FilterBase], str]] = {value: key for key, value in filters.items()}

filter_names: typing.Final[tuple[str]] = tuple(sorted(filters.keys()))
