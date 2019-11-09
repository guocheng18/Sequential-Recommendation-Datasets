from srdatasets.datasets.amazon_cds import AmazonCDs
from srdatasets.datasets.amazon_electronics import AmazonElectronics
from srdatasets.datasets.bookcrossing import BookCrossing
from srdatasets.datasets.foursquare_nyc import FourSquareNYC
from srdatasets.datasets.foursquare_tokyo import FourSquareTokyo
from srdatasets.datasets.gowalla import Gowalla
from srdatasets.datasets.lastfm1k import Lastfm1K
from srdatasets.datasets.movielens20m import MovieLens20M
from srdatasets.datasets.yelp import Yelp

_dataset_classes = {
    "Amazon-CDs": AmazonCDs,
    "Amazon-Electronics": AmazonElectronics,
    "Book-Crossing": BookCrossing,
    "FourSquare-NYC": FourSquareNYC,
    "FourSquare-Tokyo": FourSquareTokyo,
    "Gowalla": Gowalla,
    "Last.fm-1K": Lastfm1K,
    "MovieLens-20M": MovieLens20M,
    "Yelp": Yelp,
}

__datasets__ = _dataset_classes.keys()
