from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="stefan-ai",
    version="1.0.0",
    author="Radek Figiel",
    description="Local AI Photo Generator with Identity Consistency",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/figielradek7-stack/stefan",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.35.0",
        "diffusers>=0.24.0",
        "peft>=0.7.0",
        "insightface>=0.7.0",
        "pillow>=10.0.0",
        "gradio>=4.19.0",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
    ],
)
