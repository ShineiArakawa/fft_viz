"""Utility functions
"""

import logging
import os
import sys
import typing
import uuid

import numpy as np
import pydantic.config as config
import pydantic.dataclasses as dataclasses
import torch
from imgui_bundle import imgui

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Helper function for image processing

ArrayLike = typing.TypeVar('ArrayLike', np.ndarray, torch.Tensor)


def normalize_0_to_1(x: ArrayLike, based_on_min_max: bool = False) -> ArrayLike:
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

    if isinstance(x, np.ndarray):
        if based_on_min_max:
            min = np.min(x)
            max = np.max(x)
        else:
            # [-1, 1] -> [0, 1]
            min = -1.0
            max = 1.0

        return np.clip((x - min) / (max - min + 1e-8), 0.0, 1.0)

    if based_on_min_max:
        min = torch.min(x)
        max = torch.max(x)
    else:
        # [-1, 1] -> [0, 1]
        min = -1.0
        max = 1.0

    return torch.clamp((x - min) / (max - min + 1e-8), 0.0, 1.0)

# --------------------------------------------------------------------------------------------------------------------------------------------------------
# Helper function for imgui


WidgetReturn = typing.TypeVar('WidgetReturn')


def make_unique(func: typing.Callable[[typing.Any], WidgetReturn], label: str, *args, **kwargs) -> WidgetReturn:
    """Wrap the given function call with push_id/popid context to have a unique widget id.

    >>> make_unique(implot.plot_line, 'sin(2.0 * pi * x)', x, np.sin(2.0 * np.pi * x))

    Parameters
    ----------
    func : typing.Callable[[Any], WidgetReturn]
        Imgui widget function to call
    label : str
        Label of the widget

    Returns
    -------
    ret : WidgetReturn
        Returned value from `func`
    """

    unique_id = label + '_' + uuid.uuid4().hex

    imgui.push_id(unique_id)
    ret = func(label, *args, **kwargs)
    imgui.pop_id()

    return ret


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Logger utility function


def get_logger(
    log_level: str | None = None,
    name: str | None = None,
    logger_type: typing.Literal["native", "loguru"] = "loguru"
) -> logging.Logger | typing.Any:
    """Get logger object. If not found `loguru`, `logger_type` will be set to `native` automatically.
    The log level is set by `log_level` or `GLOBAL_LOG_LEVEL` environment variable.

    Parameters
    ----------
    log_level : str, optional
        Log level, by default None
    name : str, optional
        Logger name. If you use loguru, this option will be ignored, by default None
    logger_type : str, optional
        Logger type. You can select from ['native', 'loguru'], by default "loguru"

    Returns
    -------
    logging.Logger | loguru.Logger
        logger object
    """

    log_level = log_level if log_level else os.environ.get("GLOBAL_LOG_LEVEL", 'info')
    logger = None

    if logger_type == "loguru":
        try:
            import loguru
            from loguru import logger as _logger
        except ModuleNotFoundError:
            logger_type = "native"
            pass
        pass

    if logger_type == "native":
        logging.basicConfig(
            level=log_level.upper(),
            format="logging:@:%(filename)s(%(lineno)s):fn:%(funcName)s:\nlevel:%(levelname)s:%(message)s"
        )
        logger = logging.getLogger(name)
    elif logger_type == "loguru":
        _logger: loguru.Logger
        logger = _logger
        logger.remove()
        logger.add(sys.stdout, level=log_level.upper())
    else:
        raise RuntimeError(f"Unknown logger type: {logger_type}")

    return logger

# --------------------------------------------------------------------------------------------------------------------------------------------------------
# Easy paramter class with imgui widget


ValueType = typing.TypeVar('ValueType', int, float, bool)


@dataclasses.dataclass(config=config.ConfigDict(arbitrary_types_allowed=True))
class ValueEntity:
    """Offering flexible parameter handling with imgui
    """

    value: ValueType
    value_type: typing.Type[ValueType]
    min_value: typing.Optional[ValueType] = None
    max_value: typing.Optional[ValueType] = None

    id: str = None

    def __post_init__(self):
        self.id = self.id or uuid.uuid4().hex

    def _draw_slider_widget(self, name: str, hide_label: bool = False) -> ValueType:
        prefix = '## ' if hide_label else ''

        imgui.push_id(f'{name}_slider_{self.id}')

        value: ValueType = None
        if self.value_type is int:
            value = imgui.slider_int(
                prefix + name,
                self.value,
                self.min_value if self.min_value is not None else 0,
                self.max_value if self.max_value is not None else 100,
            )[1]
        elif self.value_type is float:
            value = imgui.slider_float(
                prefix + name,
                self.value,
                self.min_value if self.min_value is not None else 0.0,
                self.max_value if self.max_value is not None else 1.0,
            )[1]
        else:
            raise TypeError(f'Unsupported value type for {name}: {self.value_type}')

        imgui.pop_id()

        self.value = self.value_type(value)

        return self.value

    def _draw_input_widget(self, name: str, hide_label: bool = False) -> ValueType:
        prefix = '## ' if hide_label else ''

        imgui.push_id(f'{name}_input_{self.id}')

        value: ValueType = None
        if self.value_type is int:
            value = imgui.input_int(
                prefix + name,
                self.value,
            )[1]
        elif self.value_type is float:
            value = imgui.input_float(
                prefix + name,
                self.value,
            )[1]
        else:
            raise TypeError(f'Unsupported value type for {name}: {self.value_type}')

        imgui.pop_id()

        self.value = self.value_type(value)

        return self.value

    def _draw_checkbox_widget(self, name: str, hide_label: bool = False) -> ValueType:
        prefix = '## ' if hide_label else ''

        imgui.push_id(f'{name}_checkbox_{self.id}')

        value: ValueType = None
        if self.value_type is bool:
            value = imgui.checkbox(
                prefix + name,
                self.value,
            )[1]
        else:
            raise TypeError(f'Unsupported value type for {name}: {self.value_type}')

        imgui.pop_id()

        self.value = self.value_type(value)

        return self.value

    # ---------------------------------------------------------------------------------------------------
    # Public function

    def draw_param_widgets(self, name: str) -> ValueType:
        """Draw interactive widgets to assign parameter values."""

        if self.value_type in [float, int]:
            # Draw slider input
            imgui.text(f'{name} Slider')
            imgui.same_line()
            self._draw_slider_widget(name, hide_label=True)

            # Draw direct input widget, too.
            imgui.text(f'{name} Input')
            imgui.same_line()
            self._draw_input_widget(name, hide_label=True)
        elif self.value_type is bool:
            imgui.text(name)
            imgui.same_line()
            self._draw_checkbox_widget(name, hide_label=True)
        else:
            raise TypeError(f'Unsupported value type for {name}: {self.value_type}')

        return self.value
