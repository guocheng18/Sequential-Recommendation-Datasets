import pytest

from srdatasets.download import _download

datasets = [
    "Amazon-Apps",
    "CiteULike",
    "FourSquare-NYC",
    "Gowalla",
    "Lastfm1K",
    "MovieLens20M",
    "TaFeng",
]


@pytest.mark.run(order=1)
@pytest.mark.parametrize("name", datasets)
def test_download_datasets(name):
    _download(name)
