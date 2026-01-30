"""Setup script for the polylingo package."""

from setuptools import setup, find_packages

setup(
    name="polylingo",
    version="0.1.0",
    description="Machine Learning for Unicode Characters",
    author="Polylingo Team",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "Pillow>=9.0.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.60.0",
        "requests>=2.25.0",
    ],
)
