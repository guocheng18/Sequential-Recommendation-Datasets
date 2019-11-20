import os
from datetime import datetime

import pandas as pd

from srdatasets.datasets.dataset import Dataset
from srdatasets.datasets.utils import download_url, extract


class CiteULike(Dataset):

    __url__ = "http://konect.cc/files/download.tsv.citeulike-ut.tar.bz2"
    __corefile__ = os.path.join("citeulike-ut", "out.citeulike-ut")

    def download(self) -> None:
        filepath = self.home.joinpath("download.tsv.citeulike-ut.tar.bz2")
        download_url(self.__url__, filepath)
        extract(filepath, self.home)

    def transform(self) -> pd.DataFrame:
        df = pd.read_csv(
            self.home.joinpath(self.__corefile__),
            sep=" ",
            header=None,
            index_col=False,
            names=["user_id", "tag_id", "positive", "timestamp"],
            usecols=[0, 1, 3],
            comment="%",
        )
        df["timestamp"] = df["timestamp"].apply(int)
        df = df.rename(columns={"tag_id": "item_id"})
        return df
