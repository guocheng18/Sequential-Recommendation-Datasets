from datetime import datetime

import pandas as pd
from tqdm import tqdm

from srdatasets.datasets.dataset import Dataset
from srdatasets.datasets.utils import download_url, extract

tqdm.pandas()


class Gowalla(Dataset):

    __url__ = "https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz"
    __corefile__ = "loc-gowalla_totalCheckins.txt"

    def download(self):
        download_url(self.__url__, self.rawpath)
        extract(self.rawpath, self.rootdir.joinpath("loc-gowalla_totalCheckins.txt"))

    def transform(self):
        """ Time: yyyy-mm-ddThh:mm:ssZ -> timestamp """
        df = pd.read_csv(
            self.rootdir.joinpath(self.__corefile__),
            sep="\t",
            names=["user_id", "check_in_time", "latitude", "longtitude", "location_id"],
            usecols=[0, 1, 4],
        )
        df["check_in_time"] = df["check_in_time"].progress_apply(
            lambda x: int(datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ").timestamp())
        )
        df = df.rename(columns={"location_id": "item_id", "check_in_time": "timestamp"})
        return df
