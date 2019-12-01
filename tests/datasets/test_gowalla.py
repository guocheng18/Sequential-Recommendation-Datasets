import os

from srdatasets.datasets import Gowalla
from srdatasets.utils import __warehouse__


def test_download_and_trandform():
    rawdir = __warehouse__.joinpath("Gowalla", "raw")
    os.makedirs(rawdir, exist_ok=True)
    gowalla = Gowalla(rawdir)
    gowalla.download()
    assert rawdir.joinpath(gowalla.__corefile__).exists()
    df = gowalla.transform()
    assert all(c in df.columns for c in ["user_id", "item_id", "timestamp"])
    assert len(df) > 0
