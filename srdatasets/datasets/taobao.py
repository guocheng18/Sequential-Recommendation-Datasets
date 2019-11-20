import logging
import os

import pandas as pd

from srdatasets.datasets.dataset import Dataset
from srdatasets.datasets.utils import extract

logger = logging.getLogger(__name__)


class Taobao(Dataset):

    __url__ = "https://tianchi.aliyun.com/dataset/dataDetail?dataId=649"
    __corefile__ = "UserBehavior.csv"

    def download(self) -> None:
        if not self.home.joinpath("UserBehavior.csv.zip").exists():
            logger.warning(
                "Since Taobao dataset is not directly accessible, please visit %s and download it manually, after downloaded, place file 'UserBehavior.csv.zip' under %s and run this command again",
                self.__url__,
                self.home,
            )
        else:
            extract(self.home.joinpath("UserBehavior.csv.zip"), self.home)

    def transform(self) -> pd.DataFrame:
        df = pd.read_csv(
            self.home.joinpath(self.__corefile__),
            header=None,
            index_col=False,
            names=["user_id", "item_id", "category_id", "behavior_type", "timestamp"],
            usecols=[0, 1, 4],
        )
        return df
