from srdatasets.datasets import __datasets__
from srdatasets.download import _download


def test_download_all_datasets():
    _download("Amazon-Apps")
    _download("FourSquare-NYC")
    for d in __datasets__:
        if not d.startswith("Amazon") and not d.startswith("FourSquare")
            _download(d)
