import glob
import logging
import os
import shutil
from zipfile import ZipFile

import pandas as pd
import wget
from pandas import DataFrame

from srdatasets.datasets.dataset import Dataset

logger = logging.getLogger(__name__)


class MovieLens20M(Dataset):

    __download_url__ = "http://files.grouplens.org/datasets/movielens/ml-20m.zip"
    __corefile__ = "ratings.csv"

    def download(self) -> None:
        try:
            wget.download(self.__download_url__, out=self.home)
            logger.info("Download successful, unzippping...")
        except:
            logger.exception("Download failed, please retry")
            for f in glob.glob(self.home.joinpath("*.tmp")):
                os.remove(f)
            return

        zipfile_path = self.home.joinpath("ml-20.zip")
        unzip_folder = self.home.joinpath("ml-20")

        with ZipFile(zipfile_path) as zipObj:
            zipObj.extractall(self.home)

        shutil.move(unzip_folder.joinpath("*"), self.home)
        os.rmdir(unzip_folder)
        logger.info("Finished, dataset location: %s", self.home)

    def transform(self, rating_threshold) -> DataFrame:
        """ Records with rating less than `rating_threshold` are dropped
        """
        df = pd.read_csv(
            self.home.joinpath(self.__corefile__),
            header=0,
            names=["user_id", "item_id", "rating", "timestamp"],
        )
        df = df[df.rating >= rating_threshold].drop("rating")
        return df
