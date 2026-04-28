# Stefan - Local AI Photo Generator

**A production-ready, fully-local AI system that generates highly realistic photos of consistent AI models based on reference images.**

## 🎯 Features

- ✅ **Stable Diffusion XL** integration (state-of-the-art image generation)
- ✅ **Identity Consistency** using LoRA fine-tuning + Face embeddings (InsightFace)
- ✅ **DreamBooth** training for personalized models
- ✅ **Image-to-Image Transformation** (changing poses, outfits, environments)
- ✅ **Inpainting** (edit specific parts of images)
- ✅ **Face Swap** / Identity Preservation
- ✅ **Prompt-Based Generation** with style transfer
- ✅ **Batch Generation** (process multiple images)
- ✅ **100% Local** - No API calls, no internet required
- ✅ **GPU Acceleration** (CUDA) with CPU fallback
- ✅ **Web UI** (Gradio interface)
- ✅ **NSFW Filter** (configurable)
- ✅ **Seed Control** for reproducibility
- ✅ **Metadata Tracking**

## 📋 System Requirements

### Minimum
- **GPU**: 8GB VRAM (NVIDIA RTX 2060 or better)
- **RAM**: 16GB system RAM
- **Storage**: 50GB (for models)
- **OS**: Ubuntu 20.04+ / Windows 10+ / macOS 11+
- **Python**: 3.10+

### Recommended
- **GPU**: 12GB+ VRAM (RTX 3060 Ti or better)
- **RAM**: 32GB system RAM
- **Storage**: 100GB (SSD recommended)

### CPU-Only (Much Slower)
- Requires 32GB+ RAM
- ~30-60 seconds per image on high-end CPU

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/figielradek7-stack/stefan.git
cd stefan
```

### 2. Create Virtual Environment
```bash
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Models (Automatic)
```bash
python scripts/download_models.py
```
This will download:
- Stable Diffusion XL (8.5GB)
- SDXL Refiner (6.9GB)
- InsightFace face detector & embeddings (150MB)
- Additional safety components

### 5. Launch Web UI
```bash
python ui/app.py
```
Open browser to: `http://localhost:7860`

## 📚 Project Structure

```
stefan/
├── models/
│   ├── __init__.py
│   ├── pipeline.py           # Main inference pipeline
│   ├── lora.py               # LoRA fine-tuning logic
│   ├── dreambooth.py         # DreamBooth training
│   ├── face_embedding.py     # InsightFace integration
│   └── safety.py             # NSFW filter
├── training/
│   ├── __init__.py
│   ├── lora_train.py         # LoRA training script
│   ├── dreambooth_train.py   # DreamBooth training script
│   └── data_loader.py        # Data loading utilities
├── inference/
│   ├── __init__.py
│   ├── generator.py          # Image generation engine
│   ├── inpainter.py          # Inpainting module
│   └── batch_processor.py    # Batch generation
├── ui/
│   ├── __init__.py
│   ├── app.py                # Gradio web interface
│   └── static/               # Frontend assets
├── utils/
│   ├── __init__.py
│   ├── config.py             # Configuration management
│   ├── device.py             # GPU/CPU device handler
│   ├── image_utils.py        # Image processing
│   ├── metadata.py           # Metadata tracking
│   └── logger.py             # Logging utility
├── scripts/
│   ├── download_models.py    # Model downloader
│   ├── train_person.py       # End-to-end training script
│   └── benchmark.py          # Performance testing
├── examples/
│   ├── reference/            # Example reference images
│   ├── outputs/              # Generated images
│   ├── example_prompts.txt   # Curated prompt examples
│   └── config_examples.json  # Example configurations
├── .env.example              # Environment variables template
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup
├── Dockerfile                # Docker configuration
└── INSTALLATION_GUIDE.md     # Detailed setup guide
```

## 🎓 Training Your Own Model

### Option 1: Quick LoRA Fine-tuning (Recommended)
```bash
python scripts/train_person.py \
  --mode lora \
  --reference_images ./examples/reference/ \
  --person_name "john_doe" \
  --output_path ./models/lora/
```

### Option 2: DreamBooth Training (Better Quality, Slower)
```bash
python scripts/train_person.py \
  --mode dreambooth \
  --reference_images ./examples/reference/ \
  --person_name "john_doe" \
  --output_path ./models/dreambooth/
```

## 🖼️ Usage Examples

### 1. Generate New Photos
```python
from models.pipeline import StabilityPipeline

pipeline = StabilityPipeline()

# Generate image of a person
image = pipeline.generate(
    prompt="A professional headshot of john_doe in a business suit, studio lighting",
    num_inference_steps=50,
    guidance_scale=7.5,
    height=768,
    width=512,
    seed=42
)

image.save("output.png")
```

### 2. Image-to-Image Transformation
```python
from PIL import Image

reference = Image.open("reference.jpg")

image = pipeline.img2img(
    image=reference,
    prompt="john_doe at the beach, sunset, happy expression",
    strength=0.8,
    guidance_scale=7.5,
    seed=42
)

image.save("beach_photo.png")
```

### 3. Inpainting (Edit Specific Area)
```python
from PIL import Image

original = Image.open("photo.jpg")
mask = Image.open("mask.png")  # White = area to edit, Black = keep

image = pipeline.inpaint(
    image=original,
    mask=mask,
    prompt="john_doe wearing a red shirt",
    strength=0.8,
    guidance_scale=7.5
)

image.save("edited.png")
```

### 4. Batch Generation
```python
prompts = [
    "john_doe in a business meeting",
    "john_doe at the gym",
    "john_doe on vacation",
    "john_doe playing guitar"
]

images = pipeline.batch_generate(prompts, num_images_per_prompt=2)

for idx, img in enumerate(images):
    img.save(f"batch_{idx}.png")
```

### 5. With Face Embedding (Enhanced Identity)
```python
from models.face_embedding import FaceEmbedder

embedder = FaceEmbedder()
reference_image = Image.open("reference.jpg")

# Extract face embedding
face_embedding = embedder.extract(reference_image)

# Generate with face constraint
image = pipeline.generate_with_face(
    prompt="john_doe in a tuxedo",
    face_embedding=face_embedding,
    guidance_scale=7.5,
    face_weight=0.8  # How strongly to enforce identity
)

image.save("consistent_john.png")
```

## ⚙️ Configuration

Create `.env` file:
```env
# Device Configuration
DEVICE=cuda  # or cpu
PRECISION=fp16  # or fp32 for CPU

# Model Configuration
MODEL_ID=stabilityai/stable-diffusion-xl-base-1.0
REFINER_ID=stabilityai/stable-diffusion-xl-refiner-1.0
USE_REFINER=true

# Safety
ENABLE_NSFW_FILTER=true
NSFW_THRESHOLD=0.5

# Generation Defaults
DEFAULT_STEPS=50
DEFAULT_GUIDANCE_SCALE=7.5
DEFAULT_HEIGHT=768
DEFAULT_WIDTH=512

# Training
LORA_RANK=64
LORA_ALPHA=64
DREAMBOOTH_LEARNING_RATE=1e-4

# UI
UI_HOST=0.0.0.0
UI_PORT=7860
```

## 📊 Performance Benchmarks

| Configuration | Time/Image | VRAM Used | Quality |
|---------------|-----------|-----------|----------|
| RTX 4090 + FP16 | ~3-5s | 12GB | Excellent |
| RTX 3090 + FP16 | ~5-8s | 14GB | Excellent |
| RTX 3060 Ti + FP16 | ~10-15s | 8GB | Good |
| RTX 2060 + FP32 | ~20-30s | 6GB | Good |
| CPU (32GB RAM) | ~60-120s | Variable | Good |

## 🔧 Troubleshooting

### CUDA Out of Memory
```bash
# Use smaller resolution
python ui/app.py --height 512 --width 512

# Or use CPU with memory optimization
export DEVICE=cpu
python ui/app.py
```

### Slow Generation on CPU
- Reduce `num_inference_steps` from 50 to 30
- Use smaller image dimensions (512x512 instead of 768x512)
- Ensure 32GB+ RAM available

### Poor Identity Consistency
1. Use more reference images (5-10 for LoRA, 20+ for DreamBooth)
- Vary poses, lighting, backgrounds
- Use high-quality images (at least 512x512)
- Try increasing `face_weight` parameter

### Model Download Fails
```bash
# Manual download with HuggingFace CLI
huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0
```

## 🐳 Docker Setup

```bash
# Build image
docker build -t stefan-ai .

# Run with GPU
docker run --gpus all -p 7860:7860 -v $(pwd)/outputs:/app/outputs stefan-ai

# Access: http://localhost:7860
```

## 📝 Prompt Engineering Guide

### Best Practices
```
"[person identifier] [action/pose] [clothing] [environment] [lighting] [style]"

Example:
"john_doe wearing a business suit, standing in an office, professional photography, sharp focus, studio lighting"
```

### Negative Prompts (What to Avoid)
```
"blurry, low quality, distorted face, bad anatomy, watermark, text, artifacts"
```

### Style Modifiers
- Realistic portrait
- Studio photography
- Fashion photography
- Cinematic lighting
- Film photography
- Professional headshot

## 🎯 Advanced Features

### 1. Multi-Person Generation
Train separate LoRA models for multiple people:
```python
pipeline.set_lora("john_doe")
image1 = pipeline.generate("john_doe with sarah at a dinner")

pipeline.set_lora("sarah")
image2 = pipeline.generate("sarah smiling")
```

### 2. Style Transfer
```python
image = pipeline.style_transfer(
    image=reference_image,
    style="oil painting",
    strength=0.8
)
```

### 3. Batch Processing with Face Consistency
```python
for prompt in prompts:
    image = pipeline.generate_with_face(
        prompt=prompt,
        face_embedding=face_embedding,
        consistency_weight=0.9
    )
```

## 📦 Dependencies

- `torch` - Deep learning framework
- `transformers` - Hugging Face transformers
- `diffusers` - Stable Diffusion implementations
- `insightface` - Face detection & embedding
- `gradio` - Web UI framework
- `PIL` - Image processing
- `numpy` - Numerical computing
- `opencv-python` - Computer vision
- `peft` - LoRA implementation

## 🔒 Privacy & Safety

- ✅ **100% Local Processing** - No data leaves your machine
- ✅ **NSFW Filter** - Optional built-in safety checker
- ✅ **Model Weights** - All weights stored locally
- ✅ **No Tracking** - No telemetry or analytics
- ✅ **Open Source** - Full code transparency

## 📄 License

MIT License - See LICENSE file

## 🤝 Contributing

Contributions welcome! Please submit pull requests.

## ⚠️ Disclaimer

This tool is for creating AI models of **yourself or people who explicitly consent**. Using this to create deepfakes of real people without consent is illegal in many jurisdictions. Users are solely responsible for ensuring they have appropriate permissions and comply with applicable laws.

## 🚀 Next Steps

1. Install dependencies: `pip install -r requirements.txt`
2. Download models: `python scripts/download_models.py`
3. Train a model: `python scripts/train_person.py --reference_images ./examples/reference/`
4. Launch UI: `python ui/app.py`
5. Generate images!

## 📞 Support

For issues, questions, or suggestions, please open a GitHub issue.

---

**Built with ❤️ for creative AI photo generation**
