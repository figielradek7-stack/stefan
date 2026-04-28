import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from PIL import Image
from .logger import setup_logger

logger = setup_logger(__name__)


class MetadataTracker:
    """Track and save metadata for generated images."""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("./outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.output_dir / "metadata.jsonl"

    def add_entry(
        self,
        image_path: Path,
        prompt: str,
        negative_prompt: str = "",
        steps: int = 50,
        guidance_scale: float = 7.5,
        seed: int = -1,
        model: str = "sdxl",
        lora_name: Optional[str] = None,
        face_embedding: bool = False,
        style: str = "realistic",
        generation_time: float = 0.0,
        **kwargs
    ) -> Dict[str, Any]:
        """Add metadata entry for generated image.
        
        Args:
            image_path: Path to generated image
            prompt: Generation prompt
            negative_prompt: Negative prompt
            steps: Number of inference steps
            guidance_scale: Classifier-free guidance scale
            seed: Random seed
            model: Model name
            lora_name: LoRA model name (if used)
            face_embedding: Whether face embedding was used
            style: Image style
            generation_time: Time taken to generate
            **kwargs: Additional metadata
            
        Returns:
            Metadata dictionary
        """
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "image_path": str(image_path),
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "model": model,
            "lora_name": lora_name,
            "face_embedding": face_embedding,
            "style": style,
            "generation_time": generation_time,
            **kwargs,
        }

        # Save to JSONL file
        with open(self.metadata_file, "a") as f:
            f.write(json.dumps(metadata) + "\n")

        logger.debug(f"Metadata saved for {image_path}")
        return metadata

    def get_entries(self, limit: int = None) -> list:
        """Get metadata entries.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of metadata dictionaries
        """
        entries = []
        
        if not self.metadata_file.exists():
            return entries
        
        with open(self.metadata_file, "r") as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse metadata line {i}")
        
        return entries

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about generated images."""
        entries = self.get_entries()
        
        if not entries:
            return {"total_images": 0}
        
        total_time = sum(e.get("generation_time", 0) for e in entries)
        avg_time = total_time / len(entries) if entries else 0
        
        models = {}
        for entry in entries:
            model = entry.get("model", "unknown")
            models[model] = models.get(model, 0) + 1
        
        return {
            "total_images": len(entries),
            "total_generation_time": total_time,
            "average_generation_time": avg_time,
            "models_used": models,
            "with_lora": sum(1 for e in entries if e.get("lora_name")),
            "with_face_embedding": sum(1 for e in entries if e.get("face_embedding")),
        }
