import os

import pandas as pd

from srdatasets.datasets.dataset import Dataset
from srdatasets.datasets.utils import download_url, extract


class MovieLens20M(Dataset):

    __url__ = "http://files.grouplens.org/datasets/movielens/ml-20m.zip"
    __corefile__ = os.path.join("ml-20m", "ratings.csv")

    def download(self):
        download_url(self.__url__, self.rawpath)
        extract(self.rawpath, self.rootdir)

    def transform(self, rating_threshold):
        """ Records with rating less than `rating_threshold` are dropped
        """
        df = pd.read_csv(
            self.rootdir.joinpath(self.__corefile__),
            header=0,
            names=["user_id", "movie_id", "rating", "timestamp"],
        )
        df = df.rename(columns={"movie_id": "item_id"})
        df = df[df.rating >= rating_threshold].drop("rating", axis=1)
        return df
