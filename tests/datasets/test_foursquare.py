import os

from srdatasets.datasets import FourSquare
from srdatasets.utils import __warehouse__


def test_download_and_trandform():
    rawdir = __warehouse__.joinpath("FourSquare", "raw")
    os.makedirs(rawdir, exist_ok=True)
    foursquare = FourSquare(rawdir)
    cities = ["NYC", "Tokyo"]
    foursquare.download()
    for c in cities:
        assert rawdir.joinpath(foursquare.__corefile__[c]).exists()
    for c in cities:
        df = foursquare.transform(c)
        assert all(c in df.columns for c in ["user_id", "item_id", "timestamp"])
        assert len(df) > 0
