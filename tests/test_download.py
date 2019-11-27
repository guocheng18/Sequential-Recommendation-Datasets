from srdatasets.datasets import __datasets__
from srdatasets.download import _download


def test_download_all_datasets():
    for dataset in __datasets__:
        _download(dataset)
