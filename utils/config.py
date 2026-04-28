import os
from dotenv import load_dotenv
from pathlib import Path
import json

load_dotenv()


class Config:
    """Central configuration management for the entire system."""

    # Device Configuration
    DEVICE = os.getenv("DEVICE", "cuda")
    PRECISION = os.getenv("PRECISION", "fp16")

    # Model Configuration
    MODEL_ID = os.getenv(
        "MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0"
    )
    REFINER_ID = os.getenv(
        "REFINER_ID", "stabilityai/stable-diffusion-xl-refiner-1.0"
    )
    USE_REFINER = os.getenv("USE_REFINER", "true").lower() == "true"

    # Safety
    ENABLE_NSFW_FILTER = (
        os.getenv("ENABLE_NSFW_FILTER", "true").lower() == "true"
    )
    NSFW_THRESHOLD = float(os.getenv("NSFW_THRESHOLD", "0.5"))

    # Generation Defaults
    DEFAULT_STEPS = int(os.getenv("DEFAULT_STEPS", "50"))
    DEFAULT_GUIDANCE_SCALE = float(os.getenv("DEFAULT_GUIDANCE_SCALE", "7.5"))
    DEFAULT_HEIGHT = int(os.getenv("DEFAULT_HEIGHT", "768"))
    DEFAULT_WIDTH = int(os.getenv("DEFAULT_WIDTH", "512"))

    # LoRA Configuration
    LORA_RANK = int(os.getenv("LORA_RANK", "64"))
    LORA_ALPHA = int(os.getenv("LORA_ALPHA", "64"))
    LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", "0.1"))

    # DreamBooth Configuration
    DREAMBOOTH_LEARNING_RATE = float(
        os.getenv("DREAMBOOTH_LEARNING_RATE", "1e-4")
    )
    DREAMBOOTH_EPOCHS = int(os.getenv("DREAMBOOTH_EPOCHS", "100"))
    DREAMBOOTH_BATCH_SIZE = int(os.getenv("DREAMBOOTH_BATCH_SIZE", "2"))

    # Face Embedding
    FACE_DETECTION_CONFIDENCE = float(
        os.getenv("FACE_DETECTION_CONFIDENCE", "0.9")
    )
    FACE_EMBEDDING_DIM = int(os.getenv("FACE_EMBEDDING_DIM", "512"))

    # UI Configuration
    UI_HOST = os.getenv("UI_HOST", "0.0.0.0")
    UI_PORT = int(os.getenv("UI_PORT", "7860"))
    UI_SHARE = os.getenv("UI_SHARE", "false").lower() == "true"
    UI_DEBUG = os.getenv("UI_DEBUG", "false").lower() == "true"

    # Paths
    OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./outputs"))
    MODELS_DIR = Path(os.getenv("MODELS_DIR", "./models"))
    CACHE_DIR = Path(os.getenv("CACHE_DIR", "./cache"))

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = Path(os.getenv("LOG_FILE", "./logs/stefan.log"))

    # HuggingFace Token
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", None)

    @classmethod
    def create_directories(cls):
        """Create necessary directories."""
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_dtype(cls):
        """Get PyTorch dtype based on configuration."""
        import torch

        if cls.PRECISION == "fp16":
            return torch.float16
        elif cls.PRECISION == "fp32":
            return torch.float32
        else:
            return torch.float32

    @classmethod
    def to_dict(cls):
        """Convert configuration to dictionary."""
        return {
            "device": cls.DEVICE,
            "precision": cls.PRECISION,
            "model_id": cls.MODEL_ID,
            "refiner_id": cls.REFINER_ID,
            "use_refiner": cls.USE_REFINER,
            "nsfw_filter": cls.ENABLE_NSFW_FILTER,
            "default_steps": cls.DEFAULT_STEPS,
            "default_guidance_scale": cls.DEFAULT_GUIDANCE_SCALE,
            "default_height": cls.DEFAULT_HEIGHT,
            "default_width": cls.DEFAULT_WIDTH,
        }
