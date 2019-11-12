import logging
import os
import shutil
from zipfile import ZipFile

import pandas as pd

from srdatasets.datasets.dataset import Dataset
from srdatasets.datasets.utils import download_url

logger = logging.getLogger(__name__)


class MovieLens20M(Dataset):

    __url__ = "http://files.grouplens.org/datasets/movielens/ml-20m.zip"
    __corefile__ = os.path.join("ml-20m", "ratings.csv")

    def download(self) -> None:
        filepath = self.home.joinpath("ml-20m.zip")
        try:
            download_url(self.__url__, filepath)
            logger.info("Download successful, unzippping...")
        except:
            logger.exception("Download failed, please retry")
            if filepath.exists():
                os.remove(filepath)
            return

        with ZipFile(filepath) as zipObj:
            zipObj.extractall(self.home)

        logger.info("Finished, dataset location: %s", self.home)

    def transform(self, rating_threshold) -> pd.DataFrame:
        """ Records with rating less than `rating_threshold` are dropped
        """
        df = pd.read_csv(
            self.home.joinpath(self.__corefile__),
            header=0,
            names=["user_id", "movie_id", "rating", "timestamp"],
        )
        df = df.rename(columns={"movie_id": "item_id"})
        df = df[df.rating >= rating_threshold]
        df = df.drop("rating")
        return df
