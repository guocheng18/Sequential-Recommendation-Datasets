from srdatasets.datasets.amazon import Amazon
from srdatasets.datasets.foursquare import FourSquare
from srdatasets.datasets.gowalla import Gowalla
from srdatasets.datasets.lastfm1k import Lastfm1K
from srdatasets.datasets.movielens20m import MovieLens20M
from srdatasets.datasets.yelp import Yelp

dataset_classes = {
    "Amazon": Amazon,
    "FourSquare": FourSquare,
    "Gowalla": Gowalla,
    "Lastfm1K": Lastfm1K,
    "MovieLens20M": MovieLens20M,
    "Yelp": Yelp,
}

amazon_datasets = ["Amazon-" + c for c in Amazon.__corefile__.keys()]
foursquare_datasets = ["FourSquare-" + c for c in FourSquare.__corefile__.keys()]

__datasets__ = (
    amazon_datasets
    + foursquare_datasets
    + ["Gowalla", "Lastfm1K", "MovieLens20M", "Yelp"]
)
