from abc import ABC, abstractmethod


class Dataset(ABC):
    """ Base class of datasets, each dataset should inherit `download`
     (Implementation of downloading raw datasets) and `transform` method 
     (Transform raw data format to general data format)
    """

    @abstractmethod
    def download(self, dest: str):
        """ Download and extract raw dataset files """
        ...

    @abstractmethod
    def transform(self, *args):
        """ Transform to the general data format:
        a pd.DataFrame instance that contains three columns: 
        [user_id, item_id, timestamp]
        """
        ...
