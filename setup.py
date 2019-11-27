from setuptools import find_packages, setup

VERSION = "0.0.7"

with open("README.md", "r") as f:
    long_description = f.read()


setup(
    name="srdatasets",
    version=VERSION,
    author="Cheng Guo",
    author_email="guocheng672@gmail.com",
    description="A collection of research ready datasets for sequential recommendation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/guocheng2018/sequential-recommendation-datasets",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "pandas>=0.25.0",
        "tqdm>=4.33.0",
        "tabulate>=0.8.5",
        "numpy>=1.16.4",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
    ],
)
