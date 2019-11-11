import logging
import os
import tarfile
from datetime import datetime

import pandas as pd

from srdatasets.datasets.dataset import Dataset
from srdatasets.datasets.utils import download_url

logger = logging.getLogger(__name__)


class Lastfm1K(Dataset):

    __url__ = "http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-1K.tar.gz"
    __corefile__ = os.path.join(
        "lastfm-dataset-1K", "userid-timestamp-artid-artname-traid-traname.tsv"
    )

    def download(self) -> None:
        filepath = self.home.joinpath("lastfm-dataset-1K.tar.gz")
        try:
            download_url(self.__url__, filepath)
            logger.info("Download successful, unzipping...")
        except:
            logger.exception("Download failed, please try again")
            os.remove(filepath)
            return

        with tarfile.open(filepath) as tar:
            tar.extractall(self.home)

        logger.info("Finished, dataset location: %s", self.home)

    def transform(self, item_type="song") -> pd.DataFrame:
        """ item_type can be `artist` or `song`
        """
        df = pd.read_csv(
            self.home.joinpath(self.__corefile__),
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
            converters={
                "timestamp": lambda x: int(
                    datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ").timestamp()
                )
            },
        )
        if item_type == "song":
            df = df.drop("artist_id").rename(columns={"song_id": "item_id"})
        else:
            df = df.drop("song_id").rename(columns={"artist_id": "item_id"})
        return df
