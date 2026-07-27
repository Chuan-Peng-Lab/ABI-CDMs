"""Setup configuration for NSBI-CDMs: Neural Simulation-Based Inference of Conflict Diffusion Models."""

from setuptools import setup, find_packages

setup(
    name="nsbi-cdms",
    version="1.0.0",
    description="Neural Simulation-Based Inference of Conflict Diffusion Models (DDM, DMC, SSP, DSTP)",
    author="Wanke Pan",
    author_email="panwanke2023@gmail.com",
    url="https://github.com/Chuan-Peng-Lab/ABI-CDMs",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.26",
        "pandas>=2.0",
        "scipy>=1.10",
        "bayesflow>=2.0",
        "torch>=2.0",
        "keras>=3.0",
        "arviz>=0.18",
        "xarray>=2023.0",
        "numba>=0.58",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "scikit-learn>=1.3",
        "rsatoolbox>=0.3",
        "statsmodels>=0.14",
        "tqdm>=4.65",
        "fastkde>=1.0",
        "tables>=3.8",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Psychology",
    ],
)
