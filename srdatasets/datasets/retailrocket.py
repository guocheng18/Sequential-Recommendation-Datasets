import logging

import pandas as pd
from tqdm import tqdm

from srdatasets.datasets.dataset import Dataset
from srdatasets.datasets.utils import extract

tqdm.pandas()

logger = logging.getLogger(__name__)


class Retailrocket(Dataset):

    __corefile__ = "events.csv"

    def download(self):
        if not self.rootdir.joinpath("ecommerce-dataset.zip").exists():
            logger.warning(
                "Since RetailRocket dataset is not directly accessible, please visit https://www.kaggle.com/retailrocket/ecommerce-dataset and download it manually, after downloaded, place file 'ecommerce-dataset.zip' under %s and run this command again",
                self.rootdir,
            )
        else:
            extract(self.rootdir.joinpath("ecommerce-dataset.zip"), self.rootdir)

    def transform(self):
        df = pd.read_csv(
            self.rootdir.joinpath(self.__corefile__),
            header=0,
            index_col=False,
            usecols=[0, 1, 3],
        )
        df["timestamp"] = df["timestamp"].progress_apply(lambda x: int(x / 1000))
        df = df.rename({"visitorid": "user_id", "itemid": "item_id"})
        return df
