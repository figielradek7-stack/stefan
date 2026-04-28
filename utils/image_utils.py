import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Tuple, Optional
import torch
from .logger import setup_logger

logger = setup_logger(__name__)


def load_image(
    image_path: Union[str, Path], size: Optional[Tuple[int, int]] = None
) -> Image.Image:
    """Load image from file.
    
    Args:
        image_path: Path to image file
        size: Optional tuple (width, height) to resize
        
    Returns:
        PIL Image
    """
    image = Image.open(image_path).convert("RGB")
    
    if size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    
    logger.debug(f"Loaded image from {image_path}, size: {image.size}")
    return image


def save_image(image: Image.Image, output_path: Union[str, Path], quality: int = 95):
    """Save image to file.
    
    Args:
        image: PIL Image
        output_path: Path to save
        quality: JPEG quality (1-100)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix.lower() in [".jpg", ".jpeg"]:
        image.save(output_path, quality=quality, optimize=True)
    else:
        image.save(output_path, quality=quality)
    logger.info(f"Saved image to {output_path}")


def resize_image(
    image: Image.Image,
    width: int,
    height: int,
    keep_aspect: bool = True
) -> Image.Image:
    """Resize image with optional aspect ratio preservation.
    
    Args:
        image: PIL Image
        width: Target width
        height: Target height
        keep_aspect: If True, add padding to maintain aspect ratio
        
    Returns:
        Resized PIL Image
    """
    if keep_aspect:
        # Calculate aspect ratios
        img_aspect = image.width / image.height
        target_aspect = width / height
        
        if img_aspect > target_aspect:
            # Image is wider
            new_width = width
            new_height = int(width / img_aspect)
        else:
            # Image is taller
            new_height = height
            new_width = int(height * img_aspect)
        
        # Resize and pad
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create canvas with white background
        canvas = Image.new("RGB", (width, height), (255, 255, 255))
        offset = ((width - new_width) // 2, (height - new_height) // 2)
        canvas.paste(image, offset)
        
        return canvas
    else:
        return image.resize((width, height), Image.Resampling.LANCZOS)


def create_mask(
    image: Image.Image,
    mask_type: str = "circle",
    center: Optional[Tuple[int, int]] = None,
    radius: Optional[int] = None
) -> Image.Image:
    """Create a mask image for inpainting.
    
    Args:
        image: Input image
        mask_type: Type of mask ("circle", "square", "face")
        center: Center coordinates (x, y)
        radius: Radius of mask
        
    Returns:
        Mask image (white = edit area, black = keep area)
    """
    mask = Image.new("L", (image.width, image.height), 0)  # Black base
    pixels = mask.load()
    
    if mask_type == "circle" and center and radius:
        cx, cy = center
        for x in range(max(0, cx - radius), min(image.width, cx + radius)):
            for y in range(max(0, cy - radius), min(image.height, cy + radius)):
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                    pixels[x, y] = 255  # White
    
    elif mask_type == "square" and center and radius:
        cx, cy = center
        r = radius
        for x in range(max(0, cx - r), min(image.width, cx + r)):
            for y in range(max(0, cy - r), min(image.height, cy + r)):
                pixels[x, y] = 255
    
    logger.debug(f"Created {mask_type} mask")
    return mask


def blend_images(
    image1: Image.Image,
    image2: Image.Image,
    alpha: float = 0.5
) -> Image.Image:
    """Blend two images.
    
    Args:
        image1: First image
        image2: Second image
        alpha: Blend factor (0 = image1, 1 = image2)
        
    Returns:
        Blended image
    """
    # Ensure same size
    if image1.size != image2.size:
        image2 = image2.resize(image1.size, Image.Resampling.LANCZOS)
    
    # Convert to numpy arrays
    arr1 = np.array(image1, dtype=np.float32)
    arr2 = np.array(image2, dtype=np.float32)
    
    # Blend
    blended = arr1 * (1 - alpha) + arr2 * alpha
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    return Image.fromarray(blended)


def center_crop(image: Image.Image, size: int) -> Image.Image:
    """Center crop image to square.
    
    Args:
        image: PIL Image
        size: Target size (will be size x size square)
        
    Returns:
        Cropped image
    """
    width, height = image.size
    
    if width == height:
        if width == size:
            return image
        return image.resize((size, size), Image.Resampling.LANCZOS)
    
    # Center crop
    left = (width - size) // 2
    top = (height - size) // 2
    right = left + size
    bottom = top + size
    
    return image.crop((left, top, right, bottom))


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL image to PyTorch tensor.
    
    Args:
        image: PIL Image
        
    Returns:
        Tensor with shape (3, H, W) in range [-1, 1]
    """
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1)  # CHW
    image = image * 2 - 1  # [-1, 1]
    return image


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert PyTorch tensor to PIL image.
    
    Args:
        tensor: Tensor with shape (3, H, W) in range [-1, 1]
        
    Returns:
        PIL Image
    """
    tensor = tensor.cpu().detach()
    if tensor.dim() == 4:
        tensor = tensor[0]  # Remove batch dimension
    
    tensor = (tensor / 2 + 0.5).clamp(0, 1)  # [-1, 1] -> [0, 1]
    image = tensor.permute(1, 2, 0).numpy() * 255  # HWC
    image = image.astype(np.uint8)
    return Image.fromarray(image)
