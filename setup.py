"""
Setup script for DyHuCoG package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="dyhucog",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Dynamic Hybrid Recommender via Graph-based Cooperative Games",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/DyHuCoG",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.9b0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "pre-commit>=2.15.0",
        ],
        "docs": [
            "sphinx>=4.2.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinxcontrib-bibtex>=2.4.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "dyhucog-train=scripts.train:main",
            "dyhucog-evaluate=scripts.evaluate:main",
            "dyhucog-explain=scripts.explain:main",
        ],
    },
    include_package_data=True,
    package_data={
        "dyhucog": ["config/*.yaml"],
    },
)