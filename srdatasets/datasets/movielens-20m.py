import pandas as pd

from srdatasets._arg import get_args
from srdatasets._core import preprocess_and_save
from srdatasets._utils import __storage__

__dataset__ = "movielens-20m"
__website__ = "https://grouplens.org/datasets/movielens/20m/"

args = get_args(__dataset__)

df = pd.read_csv(
    "{}/movielens-20m/raw/ratings.csv".format(__storage__),
    header=0,
    names=["user_id", "item_id", "rating", "timestamp"],
)
df = df[df.rating.ge(args.rating_threshold)]  # rating column won't be used

preprocess_and_save(df, args, __dataset__)


from srdatasets.datasets.base import Dataset

class MovieLens20M(Dataset):

    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)

    def download(self, dest):
        return super().download(dest)

    def transform(self, *args):
        return super().transform(*args)
