import torch
from typing import Tuple
from .config import Config
from .logger import setup_logger

logger = setup_logger(__name__)


def get_device() -> torch.device:
    """Get appropriate device (GPU/CPU)."""
    if Config.DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        logger.warning("CUDA not available, falling back to CPU")
        return torch.device("cpu")


def get_device_info() -> dict:
    """Get detailed device information."""
    device = get_device()
    info = {
        "device": str(device),
        "device_type": device.type,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_total"] = torch.cuda.get_device_properties(
            0
        ).total_memory / 1e9  # GB
        info["gpu_memory_allocated"] = torch.cuda.memory_allocated(0) / 1e9
        info["gpu_memory_reserved"] = torch.cuda.memory_reserved(0) / 1e9

    return info


def setup_device() -> torch.device:
    """Initialize device and log info."""
    device = get_device()
    info = get_device_info()

    logger.info("="*50)
    logger.info("Device Configuration:")
    logger.info(f"  Device: {info['device']}")
    logger.info(f"  CUDA Available: {info['cuda_available']}")

    if torch.cuda.is_available():
        logger.info(f"  GPU: {info['gpu_name']}")
        logger.info(f"  Total VRAM: {info['gpu_memory_total']:.2f} GB")
        logger.info(f"  Precision: {Config.PRECISION}")
    else:
        logger.warning("  Running on CPU - generation will be slow!")
        logger.warning("  Recommended: 32GB+ RAM for acceptable performance")

    logger.info("="*50)

    return device


def get_dtype_and_device() -> Tuple[torch.dtype, torch.device]:
    """Get PyTorch dtype and device."""
    device = get_device()
    dtype = Config.get_dtype()
    return dtype, device


def clear_gpu_memory():
    """Clear GPU cache if available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("GPU memory cleared")


def get_memory_usage() -> dict:
    """Get current memory usage."""
    try:
        import psutil
    except ImportError:
        return {"error": "psutil not installed"}

    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()

    info = {
        "cpu_percent": cpu_percent,
        "ram_total_gb": memory.total / 1e9,
        "ram_used_gb": memory.used / 1e9,
        "ram_available_gb": memory.available / 1e9,
        "ram_percent": memory.percent,
    }

    if torch.cuda.is_available():
        info["gpu_memory_allocated_gb"] = (
            torch.cuda.memory_allocated(0) / 1e9
        )
        info["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved(0) / 1e9

    return info
