import os
from pathlib import Path

from srdatasets.datasets import _dataset_corefiles

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
    for d, filename in _dataset_corefiles.items():
        if __warehouse__.joinpath(d, "raw", filename).exists():
            D.append(d)
    return D
