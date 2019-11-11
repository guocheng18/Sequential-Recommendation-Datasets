import gzip
import logging
import os
import shutil
from datetime import datetime

import pandas as pd

from srdatasets.datasets.dataset import Dataset
from srdatasets.datasets.utils import download_url

logger = logging.getLogger(__name__)


class Gowalla(Dataset):

    __url__ = "https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz"
    __corefile__ = "loc-gowalla_totalCheckins.txt"

    def download(self) -> None:
        filepath = self.home.joinpath("/loc-gowalla_totalCheckins.txt.gz")
        try:
            download_url(self.__url__, filepath)
            logger.info("Download successful, unzipping...")
        except:
            logger.exception("Download failed, please try again")
            os.remove(filepath)
            return

        with gzip.open(filepath, "rb") as f_in:
            with open(
                self.home.joinpath("loc-gowalla_totalCheckins.txt"), "w"
            ) as f_out:
                shutil.copyfileobj(f_in, f_out)

        logger.info("Finished, dataset location: %s", self.home)

    def transform(self) -> pd.DataFrame:
        """ Time: yyyy-mm-ddThh:mm:ssZ -> timestamp """
        df = pd.read_csv(
            self.home.joinpath(self.__corefile__),
            sep="\t",
            names=["user_id", "check_in_time", "latitude", "longtitude", "location_id"],
            usecols=[0, 1, 4],
            converters={
                "check_in_time": lambda x: int(
                    datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ").timestamp()
                )
            },
        )
        df = df.rename(columns={"location_id": "item_id", "check_in_time": "timestamp"})
        return df
