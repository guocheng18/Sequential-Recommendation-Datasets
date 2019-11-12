import logging
import os
from datetime import datetime
from zipfile import ZipFile

import pandas as pd

from srdatasets.datasets.dataset import Dataset
from srdatasets.datasets.utils import download_url

logger = logging.getLogger(__name__)


class FourSquare(Dataset):

    __url__ = "http://www-public.imtbs-tsp.eu/~zhang_da/pub/dataset_tsmc2014.zip"

    __corefile__ = {
        "NYC": os.path.join("dataset_tsmc2014", "dataset_TSMC2014_NYC.txt"),
        "Tokyo": os.path.join("dataset_tsmc2014", "dataset_TSMC2014_TKY.txt"),
    }

    def download(self) -> None:
        filepath = self.home.joinpath("dataset_tsmc2014.zip")
        try:
            download_url(self.__url__, filepath)
            logger.info("Download successful, unzipping...")
        except:
            logger.exception("Download failed, please try again")
            if filepath.exists():
                os.remove(filepath)
            return

        with ZipFile(filepath) as zipObj:
            zipObj.extractall(self.home)

        logger.info("Finished, dataset location: %s", self.home)

    def transform(self, city) -> pd.DataFrame:
        """ city: `NYC` or `Tokyo`
        """
        df = pd.read_csv(
            self.home.joinpath(self.__corefile__[city]),
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
            converters={
                "utc_time": lambda x: int(
                    datetime.strptime(x, "%a %b %d %H:%M:%S %z %Y").timestamp()
                )
            },
        )
        df = df.rename(columns={"venue_id": "item_id", "utc_time": "timestamp"})
        return df
