import gzip
import os
import shutil

import pandas as pd
import wget
from pandas import DataFrame

from srdatasets.datasets.base import Dataset


class Gowalla(Dataset):

    __url__ = "https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz"

    def download(self) -> None:
        wget.download(self.__url__, out=self.home)

        _zipped_file = self.home.joinpath("loc-gowalla_totalCheckins.txt.gz")
        _unzipped_file = self.home.joinpath("loc-gowalla_totalCheckins.txt")

        with gzip.open(_zipped_file, "rb") as f_in:
            with open(_unzipped_file, "w") as f_out:
                shutil.copyfileobj(f_in, f_out)

        os.remove(_zipped_file)

    def transform(self, *args) -> DataFrame:
        pass


# TODO: transform and test
