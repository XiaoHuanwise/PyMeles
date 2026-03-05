from typing import Tuple

import numpy as np
import torch


def np2tc(x: np.ndarray, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Convert NumPy array to non-autograd PyTorch tensor.

    Args:
        x (np.ndarray): NumPy array
        device (torch.device): Computing device

    Returns:
        torch.Tensor: PyTorch tensor
    """
    return torch.from_numpy(x).requires_grad_(False).to(device, non_blocking=True)


def zerostorch(s: Tuple[int], device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Generate a new zero torch.Tensor with shape and device.

    Args:
        s (tuple): Shape of the tensor
        device (torch.device): Computing device

    Returns:
        torch.Tensor: New zero tensor
    """
    return torch.zeros(s, dtype=torch.float64).requires_grad_(False).to(device, non_blocking=True)


def emptytorch(s: Tuple[int], device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Generate a new empty torch.Tensor with shape and device.

    Args:
        s (tuple): Shape of the tensor
        device (torch.device): Computing device

    Returns:
        torch.Tensor: New empty tensor
    """
    return torch.empty(s, dtype=torch.float64).requires_grad_(False).to(device, non_blocking=True)
