import glob
import logging
import os
import tarfile
from datetime import datetime

import pandas as pd
import wget

from srdatasets.datasets.dataset import Dataset

logger = logging.getLogger(__name__)


class Lastfm1K(Dataset):

    __url__ = "http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-1K.tar.gz"
    __corefile__ = os.path.join(
        "lastfm-dataset-1K", "userid-timestamp-artid-artname-traid-traname.tsv"
    )

    def download(self) -> None:
        try:
            wget.download(self.__url__, out=self.home.as_posix())
            logger.info("Download successful, unzipping...")
        except:
            logger.exception("Download failed, please retry")
            for f in glob.glob(
                os.path.join(os.getcwd(), "lastfm-dataset-1K.tar.gz*.tmp")
            ):
                os.remove(f)
            return

        zipfile_path = self.home.joinpath("lastfm-dataset-1K.tar.gz")
        with tarfile.open(zipfile_path) as tar:
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
            index_col=False,
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
