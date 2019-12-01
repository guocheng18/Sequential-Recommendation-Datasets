import os

from srdatasets.datasets import Amazon
from srdatasets.utils import __warehouse__


def test_download_and_trandform():
    rawdir = __warehouse__.joinpath("Amazon", "raw")
    os.makedirs(rawdir, exist_ok=True)
    amazon = Amazon(rawdir)
    category = "pet"
    amazon.download(category)
    assert rawdir.joinpath(amazon.__corefile__[category]).exists()
    df = amazon.transform(category, 4)
    assert all(c in df.columns for c in ["user_id", "item_id", "timestamp"])
    assert len(df) > 0
