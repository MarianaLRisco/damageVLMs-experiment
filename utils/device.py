"""Device utility functions."""

import torch


def get_device(requested: str = "cuda") -> str:
    """
    Get the best available device.

    Args:
        requested: Preferred device ("cuda", "mps", or "cpu")

    Returns:
        Actual device to use
    """
    if requested == "cuda":
        if torch.cuda.is_available():
            print(f"[INFO] Using CUDA: {torch.cuda.get_device_name(0)}")
            return "cuda"
        elif torch.backends.mps.is_available():
            print("[INFO] CUDA not available, using MPS (Apple Silicon)")
            print("[INFO] MPS fallback enabled for unsupported ops")
            return "mps"
        else:
            print("[INFO] CUDA not available, falling back to CPU")
            return "cpu"
    return requested
