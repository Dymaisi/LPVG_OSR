"""
LPVG-OSR: Limited Penetrable Visibility Graph for Open Set Recognition
"""

from setuptools import setup, find_packages

setup(
    name="lpvg_osr",
    version="0.1.0",
    description="Open Set Recognition using Limited Penetrable Visibility Graph",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "torch>=1.10.0",
        "numba>=0.55.0",
        "networkx>=2.6.0",
        "scikit-learn>=1.0.0",
        "hydra-core>=1.2.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.62.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
