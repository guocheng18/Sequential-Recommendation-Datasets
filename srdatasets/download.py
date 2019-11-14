import os

from srdatasets.utils import __warehouse__
from srdatasets.datasets import dataset_classes


def _download(dataset_name):
    _rawdir = __warehouse__.joinpath(dataset_name, "raw")
    os.makedirs(_rawdir, exist_ok=True)

    if dataset_name.startswith("Amazon"):
        dataset_classes["Amazon"](_rawdir).download(dataset_name.split("-")[1])
    elif dataset_name.startswith("FourSquare"):
        dataset_classes["FourSquare"](_rawdir).download()
    else:
        dataset_classes[dataset_name](_rawdir).download()
