import os

from srdatasets.datasets import Lastfm1K
from srdatasets.utils import __warehouse__


def test_download_and_trandform():
    rawdir = __warehouse__.joinpath("Lastfm1K", "raw")
    os.makedirs(rawdir, exist_ok=True)
    lastfm1k = Lastfm1K(rawdir)
    lastfm1k.download()
    assert rawdir.joinpath(lastfm1k.__corefile__).exists()
    item_types = ["song", "artist"]
    for it in item_types:
        df = lastfm1k.transform(it)
        assert all(c in df.columns for c in ["user_id", "item_id", "timestamp"])
        assert len(df) > 0
