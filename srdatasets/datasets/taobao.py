import logging
import os

import pandas as pd

from srdatasets.datasets.dataset import Dataset
from srdatasets.datasets.utils import extract

logger = logging.getLogger(__name__)


class Taobao(Dataset):

    __corefile__ = "UserBehavior.csv"

    def download(self):
        if not self.rootdir.joinpath("UserBehavior.csv.zip").exists():
            logger.warning(
                "Since Taobao dataset is not directly accessible, please visit https://tianchi.aliyun.com/dataset/dataDetail?dataId=649 and download it manually, after downloaded, place file 'UserBehavior.csv.zip' under %s and run this command again",
                self.rootdir,
            )
        else:
            extract(self.rootdir.joinpath("UserBehavior.csv.zip"), self.rootdir)

    def transform(self):
        df = pd.read_csv(
            self.rootdir.joinpath(self.__corefile__),
            header=None,
            index_col=False,
            names=["user_id", "item_id", "category_id", "behavior_type", "timestamp"],
            usecols=[0, 1, 4],
        )
        return df
