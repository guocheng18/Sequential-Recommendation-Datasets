import os
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from srdatasets.datasets.dataset import Dataset
from srdatasets.datasets.utils import download_url, extract

tqdm.pandas()


class CiteULike(Dataset):

    __url__ = "http://konect.cc/files/download.tsv.citeulike-ut.tar.bz2"
    __corefile__ = os.path.join("citeulike-ut", "out.citeulike-ut")

    def download(self):
        download_url(self.__url__, self.rawpath)
        extract(self.rawpath, self.rootdir)

    def transform(self):
        df = pd.read_csv(
            self.rootdir.joinpath(self.__corefile__),
            sep=" ",
            header=None,
            index_col=False,
            names=["user_id", "tag_id", "positive", "timestamp"],
            usecols=[0, 1, 3],
            comment="%",
        )
        df["timestamp"] = df["timestamp"].progress_apply(int)
        df = df.rename(columns={"tag_id": "item_id"})
        return df
