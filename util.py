import logging
import os
import sys
import typing

import numpy as np
import torch

ArrayLike = typing.TypeVar('ArrayLike', np.ndarray, torch.Tensor)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------


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

    if based_on_min_max:
        if x.ndim == 3:
            flattend = x.reshape(-1, x.shape[-1])
            min = flattend.min(dim=0, keepdim=True).values.unsqueeze(0)
            max = flattend.max(dim=0, keepdim=True).values.unsqueeze(0)
        elif x.ndim == 2:
            min = torch.min(x)
            max = torch.max(x)
        else:
            raise ValueError("Input tensor must be 2D or 3D.")
    else:
        # [-1, 1] -> [0, 1]
        min = -1.0
        max = 1.0

    return torch.clamp((x - min) / (max - min + 1e-8), 0.0, 1.0)

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
