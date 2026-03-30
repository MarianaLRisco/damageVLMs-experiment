"""Data utility functions."""

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


def get_num_workers(device: str) -> int:
    """
    Get optimal number of workers for DataLoader based on device.

    Args:
        device: Current device string

    Returns:
        Number of workers for DataLoader
    """
    if device == "cuda":
        return 4
    if device == "mps":
        return 8
    return 0


def get_batch_size(requested: int, device: str) -> int:
    """
    Adjust batch size based on device constraints.

    Args:
        requested: Desired batch size
        device: Current device string

    Returns:
        Actual batch size to use
    """
    if device == "cpu":
        return min(requested, 2)
    return requested


def get_num_workers_fuselip(device: str) -> int:
    """
    Get num_workers for FuseLIP models.

    FuseLIP with MPS (Apple Silicon) has known deadlock issues with num_workers > 0.
    Always use 0 workers (single-process data loading) for FuseLIP on MPS.

    Args:
        device: Current device string

    Returns:
        Number of workers (always 0 for MPS, 4 for CUDA)
    """
    if device == "mps":
        print("[INFO] FuseLIP: Using num_workers=0 on MPS to avoid deadlocks")
        return 0
    if device == "cuda":
        return 4
    return 0


def build_description_map(classes, prompts):
    """
    Build {class_name: prompt} dict from parallel lists.

    Args:
        classes: List of class names
        prompts: List of prompts (same order as classes)

    Returns:
        Dictionary mapping class names to prompts
    """
    return dict(zip(classes, prompts))
