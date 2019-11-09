import glob
import logging
import os
from datetime import datetime

import pandas as pd
import wget

from srdatasets.datasets.dataset import Dataset

logger = logging.getLogger(__name__)


class AmazonElectronics(Dataset):

    __url__ = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Electronics.csv"
    __corefile__ = "ratings_Electronics.csv"

    def download(self) -> None:
        try:
            wget.download(self.__url__, out=self.home.as_posix())
            logger.info("Finished, dataset location: %s", self.home)
        except:
            logger.exception("Download failed, please retry")
            for f in glob.glob(
                os.path.join(os.getcwd(), "ratings_Electronics.csv*.tmp")
            ):
                os.remove(f)

    def transform(self) -> pd.DataFrame:
        pass
