from srdatasets.datasets.gowalla import Gowalla
from srdatasets.datasets.lastfm360k import Lastfm360K
from srdatasets.datasets.movielens20m import MovieLens20M

_dataset_classes = {
    "movielens-20m": MovieLens20M,
    "lastfm-360k": Lastfm360K,
    "gowalla": Gowalla,
}

_dataset_corefiles = {
    "movielens-20m": MovieLens20M().__corefile__,
    "lastfm-360k": Lastfm360K().__corefile__,
    "gowalla": Gowalla().__corefile__,
}

__datasets__ = _dataset_classes.keys()
