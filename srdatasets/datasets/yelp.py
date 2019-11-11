import logging
import os
import tarfile
from datetime import datetime

import pandas as pd

from srdatasets.datasets.dataset import Dataset

logger = logging.getLogger(__name__)


class Yelp(Dataset):

    __corefile__ = "review.json"

    def download(self) -> None:
        if not self.home.joinpath("yelp_dataset.tar.gz").exists():
            logger.warning(
                "Since Yelp dataset is not directly accessible, please visit https://www.yelp.com/dataset/download and download it manually, after downloaded, place file 'yelp_dataset.tar.gz' under {} and run this command again".format(
                    self.home
                )
            )
        else:
            with tarfile.open(self.home.joinpath("yelp_dataset.tar.gz")) as tar:
                tar.extractall(self.home)
            logger.info("Dataset ready")

    def transform(self, stars_threshold) -> pd.DataFrame:
        df = pd.read_json(
            self.home.joinpath(self.__corefile__), orient="records", lines=True
        )
        df = df[["user_id", "business_id", "stars", "date"]]
        df["date"] = df["date"].map(
            lambda x: int(datetime.strptime(x, "%Y-%m-%d").timestamp())
        )
        df = df[df["stars"] >= stars_threshold]
        df = df.drop("stars")
        df = df.rename(columns={"business_id": "item_id", "date": "timestamp"})
        return df
