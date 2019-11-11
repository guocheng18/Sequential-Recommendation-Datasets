import os
from pathlib import Path

from srdatasets.datasets import dataset_classes
from srdatasets.datasets import __datasets__

__warehouse__ = Path(os.path.expanduser("~")).joinpath(".srdatasets")


def _get_processed_datasets():
    P = [
        "processed/dev/train.pkl",
        "processed/dev/test.pkl",
        "processed/test/train.pkl",
        "processed/test/test.pkl",
    ]
    D = []
    for d in __warehouse__.iterdir():
        if all([__warehouse__.joinpath(d, p).exists() for p in P]):
            D.append(d.stem)
    return D


def _get_downloaded_datasets():
    """ Simple check based on the existences of corefiles 
    """
    D = []
    for d in __datasets__:
        if "-" in d:
            corefile = dataset_classes[d.split("-")[0]].__corefile__[d.split("-")[1]]
        else:
            corefile = dataset_classes[d].__corefile__
        if __warehouse__.joinpath(d, "raw", corefile).exists():
            D.append(d)
    return D
