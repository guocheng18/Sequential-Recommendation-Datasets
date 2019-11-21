import logging
import os
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from srdatasets.datasets.dataset import Dataset
from srdatasets.datasets.utils import extract

tqdm.pandas()

logger = logging.getLogger(__name__)


class Tmall(Dataset):

    __corefile__ = os.path.join("data_format1", "user_log_format1.csv")

    def download(self):
        if not self.rootdir.joinpath("data_format1.zip").exists():
            logger.warning(
                "Since Tmall dataset is not directly accessible, please visit https://tianchi.aliyun.com/dataset/dataDetail?dataId=47 and download it manually, after downloaded, place file 'data_format1.zip' under %s and run this command again",
                self.rootdir,
            )
        else:
            extract(self.rootdir.joinpath("data_format1.zip"), self.rootdir)

    def transform(self):
        df = pd.read_csv(
            self.rootdir.joinpath(self.__corefile__),
            header=0,
            index_col=False,
            usecols=[0, 1, 5],
            dtype={"time_stamp": str},
        )
        df["time_stamp"] = df["time_stamp"].progress_apply(
            lambda x: int(datetime.strptime("2015" + x, "%Y%m%d").timestamp())
        )
        df = df.rename(columns={"time_stamp": "timestamp"})
        return df
