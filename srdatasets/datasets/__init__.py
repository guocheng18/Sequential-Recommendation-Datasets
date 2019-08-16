from srdatasets.datasets.gowalla import Gowalla
from srdatasets.datasets.lastfm360k import Lastfm360K
from srdatasets.datasets.movielens20m import MovieLens20M

_dataset_classes = {
    "movielens-20m": MovieLens20M,
    "lastfm-360k": Lastfm360K,
    "gowalla": Gowalla,
}

_dataset_corefiles = {
    "movielens-20m": "ratings.csv",
    "lastfm-360k": "usersha1-artmbid-artname-plays.tsv",
    "gowalla": "loc-gowalla_totalCheckins.txt",
}

__datasets__ = _dataset_classes.keys()