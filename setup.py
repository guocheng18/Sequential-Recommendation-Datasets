import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="srdatasets",
    version="0.0.3",
    author="Cheng Guo",
    author_email="guocheng672@gmail.com",
    description="A collection of research ready datasets for sequential recommendation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/guocheng2018/sequential-recommendation-datasets",
    packages=setuptools.find_packages(),
    python_requires=">=3.5",
    install_requires=[
        "pandas>=0.25.0",
        "tqdm>=4.33.0",
        "tabulate>=0.8.5",
        "numpy>=1.16.4",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
