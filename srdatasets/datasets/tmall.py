import logging
import os
from datetime import datetime

import pandas as pd

from srdatasets.datasets.dataset import Dataset
from srdatasets.datasets.utils import extract

logger = logging.getLogger(__name__)


class Tmall(Dataset):

    __url__ = "https://tianchi.aliyun.com/dataset/dataDetail?dataId=47"
    __corefile__ = os.path.join("data_format1", "user_log_format1.csv")

    def download(self) -> None:
        if not self.home.joinpath("data_format1.zip").exists():
            logger.warning(
                "Since Tmall dataset is not directly accessible, please visit %s and download it manually, after downloaded, place file 'data_format1.zip' under %s and run this command again",
                self.__url__,
                self.home,
            )
        else:
            extract(self.home.joinpath("data_format1.zip"), self.home)

    def transform(self) -> pd.DataFrame:
        df = pd.read_csv(
            self.home.joinpath(self.__corefile__),
            header=0,
            index_col=False,
            usecols=[0, 1, 5],
            converters={
                "time_stamp": lambda x: int(
                    datetime.strftime("2015" + x, "%Y%m%d").timestamp()
                )
            },
        )
        df = df.rename(columns={"time_stamp": "timestamp"})
        return df
