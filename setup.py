"""
Setup script for AI/ML Pipeline Workshop - Synthetic Data Generation
Makes DataCollection/src/ importable as a package
"""

from setuptools import setup, find_packages

setup(
    name="synthetic-data-pipeline",
    version="1.0.0",
    description="Synthetic data generation pipeline for social movement analysis",
    author="Laura Dozal",
    author_email="",
    license="GNU GPL v3",
    packages=find_packages(where="DataCollection"),
    package_dir={"": "DataCollection"},
    python_requires=">=3.8",
    install_requires=[
        "google-generativeai>=0.3.0",
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "Pillow>=9.5.0",
        "opencv-python>=4.7.0",
        "python-dotenv>=1.0.0",
        "PyYAML>=6.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.0",
        "lxml>=4.9.0",
        "jsonschema>=4.17.0",
        "ipywidgets>=8.0.0",
        "python-dateutil>=2.8.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
