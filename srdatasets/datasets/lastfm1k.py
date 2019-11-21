import os
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from srdatasets.datasets.dataset import Dataset
from srdatasets.datasets.utils import download_url, extract

tqdm.pandas()


class Lastfm1K(Dataset):

    __url__ = "http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-1K.tar.gz"
    __corefile__ = os.path.join(
        "lastfm-dataset-1K", "userid-timestamp-artid-artname-traid-traname.tsv"
    )

    def download(self):
        download_url(self.__url__, self.rawpath)
        extract(self.rawpath, self.rootdir)

    def transform(self, item_type):
        """ item_type can be `artist` or `song`
        """
        df = pd.read_csv(
            self.rootdir.joinpath(self.__corefile__),
            sep="\t",
            names=[
                "user_id",
                "timestamp",
                "artist_id",
                "artist_name",
                "song_id",
                "song_name",
            ],
            usecols=[0, 1, 2, 4],
        )
        df["timestamp"] = df["timestamp"].progress_apply(
            lambda x: int(datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ").timestamp())
        )
        if item_type == "song":
            df = df.drop("artist_id", axis=1).rename(columns={"song_id": "item_id"})
        else:
            df = df.drop("song_id", axis=1).rename(columns={"artist_id": "item_id"})
        return df
