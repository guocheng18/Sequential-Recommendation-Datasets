from srdatasets.datasets.gowalla import Gowalla
from srdatasets.datasets.lastfm1k import Lastfm1K
from srdatasets.datasets.movielens20m import MovieLens20M

_dataset_classes = {
    "MovieLens-20M": MovieLens20M,
    "Last.fm-1K": Lastfm1K,
    "Gowalla": Gowalla,
}

_dataset_corefiles = {
    "MovieLens-20M": MovieLens20M().__corefile__,
    "Last.fm-1K": Lastfm1K().__corefile__,
    "Gowalla": Gowalla().__corefile__,
}

__datasets__ = _dataset_classes.keys()
