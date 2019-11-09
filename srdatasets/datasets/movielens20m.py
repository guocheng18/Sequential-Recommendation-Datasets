import glob
import logging
import os
import shutil
from zipfile import ZipFile

import pandas as pd
import wget

from srdatasets.datasets.dataset import Dataset

logger = logging.getLogger(__name__)


class MovieLens20M(Dataset):

    __url__ = "http://files.grouplens.org/datasets/movielens/ml-20m.zip"
    __corefile__ = os.path.join("ml-20m", "ratings.csv")

    def download(self) -> None:
        try:
            wget.download(self.__url__, out=self.home.as_posix())
            logger.info("Download successful, unzippping...")
        except:
            logger.exception("Download failed, please retry")
            for f in glob.glob(os.path.join(os.getcwd(), "ml-20m.zip*.tmp")):
                os.remove(f)
            return

        zipfile_path = self.home.joinpath("ml-20m.zip")
        with ZipFile(zipfile_path) as zipObj:
            zipObj.extractall(self.home)

        logger.info("Finished, dataset location: %s", self.home)

    def transform(self, rating_threshold) -> pd.DataFrame:
        """ Records with rating less than `rating_threshold` are dropped
        """
        df = pd.read_csv(
            self.home.joinpath(self.__corefile__),
            header=0,
            names=["user_id", "item_id", "rating", "timestamp"],
        )
        df = df[df.rating >= rating_threshold].drop("rating")
        return df
