import logging
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from srdatasets.datasets.dataset import Dataset
from srdatasets.datasets.utils import extract

tqdm.pandas()

logger = logging.getLogger(__name__)


class Yelp(Dataset):

    __corefile__ = "review.json"

    def download(self):
        if not self.rootdir.joinpath("yelp_dataset.tar.gz").exists():
            logger.warning(
                "Since Yelp dataset is not directly accessible, please visit https://www.yelp.com/dataset/download and download it manually, after downloaded, place file 'yelp_dataset.tar.gz' under %s and run this command again",
                self.rootdir,
            )
        else:
            extract(self.rootdir.joinpath("yelp_dataset.tar.gz"), self.rootdir)

    def transform(self, stars_threshold):
        df = pd.read_json(
            self.rootdir.joinpath(self.__corefile__), orient="records", lines=True
        )
        df = df[["user_id", "business_id", "stars", "date"]]
        df["date"] = df["date"].progress_apply(
            lambda x: int(datetime.strptime(x, "%Y-%m-%d").timestamp())
        )
        df = df[df["stars"] >= stars_threshold].drop("stars", axis=1)
        df = df.rename(columns={"business_id": "item_id", "date": "timestamp"})
        return df
