import os
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from srdatasets.datasets.dataset import Dataset
from srdatasets.datasets.utils import download_url, extract

tqdm.pandas()


class FourSquare(Dataset):

    __url__ = "http://www-public.imtbs-tsp.eu/~zhang_da/pub/dataset_tsmc2014.zip"

    __corefile__ = {
        "NYC": os.path.join("dataset_tsmc2014", "dataset_TSMC2014_NYC.txt"),
        "Tokyo": os.path.join("dataset_tsmc2014", "dataset_TSMC2014_TKY.txt"),
    }

    def download(self):
        download_url(self.__url__, self.rawpath)
        extract(self.rawpath, self.rootdir)

    def transform(self, city):
        """ city: `NYC` or `Tokyo`
        """
        df = pd.read_csv(
            self.rootdir.joinpath(self.__corefile__[city]),
            sep="\t",
            header=None,
            names=[
                "user_id",
                "venue_id",
                "venue_category_id",
                "venue_category_name",
                "latitude",
                "longtitude",
                "timezone_offset",
                "utc_time",
            ],
            usecols=[0, 1, 7],
        )
        df["utc_time"] = df["utc_time"].progress_apply(
            lambda x: int(datetime.strptime(x, "%a %b %d %H:%M:%S %z %Y").timestamp())
        )
        df = df.rename(columns={"venue_id": "item_id", "utc_time": "timestamp"})
        return df
