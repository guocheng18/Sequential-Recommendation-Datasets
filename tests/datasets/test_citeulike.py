import os

from srdatasets.datasets import CiteULike
from srdatasets.utils import __warehouse__


def test_download_and_trandform():
    rawdir = __warehouse__.joinpath("CiteULike", "raw")
    os.makedirs(rawdir, exist_ok=True)
    citeulike = CiteULike(rawdir)
    citeulike.download()
    assert rawdir.joinpath(citeulike.__corefile__).exists()
    df = citeulike.transform()
    assert all(c in df.columns for c in ["user_id", "item_id", "timestamp"])
    assert len(df) > 0
