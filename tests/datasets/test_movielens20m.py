import os

from srdatasets.datasets import MovieLens20M
from srdatasets.utils import __warehouse__


def test_download_and_trandform():
    rawdir = __warehouse__.joinpath("MovieLens20M", "raw")
    os.makedirs(rawdir, exist_ok=True)
    movielens20m = MovieLens20M(rawdir)
    movielens20m.download()
    assert rawdir.joinpath(movielens20m.__corefile__).exists()
    df = movielens20m.transform(4)
    assert all(c in df.columns for c in ["user_id", "item_id", "timestamp"])
    assert len(df) > 0
