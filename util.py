import typing

import numpy as np
import torch

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
