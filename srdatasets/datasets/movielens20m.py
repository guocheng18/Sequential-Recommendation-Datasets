import os
import shutil
from zipfile import ZipFile

import pandas as pd
import wget
from pandas import DataFrame

from srdatasets.datasets.base import Dataset


class MovieLens20M(Dataset):

    __url__ = "http://files.grouplens.org/datasets/movielens/ml-20m.zip"

    def download(self) -> None:  # ../raw/
        wget.download(self.__url__, out=self.home)

        _zipped_file = self.home.joinpath("ml-20.zip")
        _unzipped_folder = self.home.joinpath("ml-20")

        with ZipFile(_zipped_file) as zipObj:
            zipObj.extractall(self.home)

        shutil.move(os.path.join(_unzipped_folder, "*"), self.home)
        os.rmdir(_unzipped_folder)
        os.remove(_zipped_file)

    def transform(self, rating_threshold) -> DataFrame:
        df = pd.read_csv(
            self.home.joinpath("ratings.csv"),
            header=0,
            names=["user_id", "item_id", "rating", "timestamp"],
        )
        df = df[df.rating.ge(rating_threshold)].drop("rating")
        return df

# TODO: test