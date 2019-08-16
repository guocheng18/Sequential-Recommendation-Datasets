from pathlib import Path

from pandas import DataFrame


class Dataset(object):
    """ Base class of datasets, each dataset should inherit `download`
     (implementation of downloading raw datasets) and `transform` method 
     (transforming raw data format to general data format)
    """

    def __init__(self, home: Path):
        """ `home` is local path of the raw dataset """
        self.home = home

    def download(self) -> None:
        """ Download and extract raw dataset files """
        raise NotImplementedError

    def transform(self, *args) -> DataFrame:
        """ Transform to the general data format, which is
        a pd.DataFrame instance that contains three columns: [user_id, item_id, timestamp]
        """
        raise NotImplementedError
