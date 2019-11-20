import os
from datetime import datetime

import pandas as pd

from srdatasets.datasets.dataset import Dataset
from srdatasets.datasets.utils import download_url, extract


class TaFeng(Dataset):

    __url__ = "https://sites.google.com/site/dataminingcourse2009/spring2016/annoucement2016/assignment3/D11-02.zip"
    __corefile__ = ["D11", "D12", "D01", "D02"]

    def download(self) -> None:
        filepath = self.home.joinpath("D11-02.ZIP")
        download_url(self.__url__, filepath)
        extract(filepath, self.home)

    def transform(self) -> pd.DataFrame:
        dfs = []
        for cf in self.__corefile__:
            df = pd.read_csv(
                self.home.joinpath(cf),
                sep=";",
                header=0,
                index_col=False,
                names=[
                    "timestamp",
                    "user_id",
                    "age",
                    "area",
                    "pcate",
                    "item_id",
                    "number",
                    "cost",
                    "price",
                ],
                usecols=[0, 1, 5],
                encoding="big5",
                converters={
                    "timestamp": lambda x: int(
                        datetime.strptime(x, "%Y-%m-%d %H:%M:%S").timestamp()
                    )
                },
            )
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)
