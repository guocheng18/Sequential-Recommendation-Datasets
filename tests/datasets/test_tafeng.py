import os

from srdatasets.datasets import TaFeng
from srdatasets.utils import __warehouse__


def test_download_and_trandform():
    rawdir = __warehouse__.joinpath("TaFeng", "raw")
    os.makedirs(rawdir, exist_ok=True)
    tafeng = TaFeng(rawdir)
    tafeng.download()
    assert all(rawdir.joinpath(cf).exists() for cf in tafeng.__corefile__)
    df = tafeng.transform()
    assert all(c in df.columns for c in ["user_id", "item_id", "timestamp"])
    assert len(df) > 0
