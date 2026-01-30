"""Device detection utilities."""

import torch


def get_device(device_str: str = "auto") -> torch.device:
    """Get the best available device.

    Args:
        device_str: Device specification. One of "auto", "cuda", "mps", "cpu".

    Returns:
        torch.device for the selected device.
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def should_pin_memory(device: torch.device) -> bool:
    """Check if pin_memory should be used for this device.

    Args:
        device: The torch device.

    Returns:
        True if pin_memory should be enabled.
    """
    # MPS does not support pin_memory
    return device.type == "cuda"
