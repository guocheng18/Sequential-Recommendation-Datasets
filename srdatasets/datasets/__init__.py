from srdatasets.datasets.gowalla import Gowalla
from srdatasets.datasets.lastfm1k import Lastfm1K
from srdatasets.datasets.movielens20m import MovieLens20M

_dataset_classes = {
    "movielens-20m": MovieLens20M,
    "lastfm-1k": Lastfm1K,
    "gowalla": Gowalla,
}

_dataset_corefiles = {
    "movielens-20m": MovieLens20M().__corefile__,
    "lastfm-1k": Lastfm1K().__corefile__,
    "gowalla": Gowalla().__corefile__,
}

__datasets__ = _dataset_classes.keys()
