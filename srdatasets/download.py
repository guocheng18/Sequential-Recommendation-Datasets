import os

from srdatasets.utils import __warehouse__
from srdatasets.datasets import _dataset_classes


def _download(dataset_name: str) -> None:
    _rawdir = __warehouse__.joinpath(dataset_name, "raw")
    os.makedirs(_rawdir, exist_ok=True)

    _dataset_classes[dataset_name](_rawdir).download()
