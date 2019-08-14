import math
import os
import pickle
import random

from srdatasets.utils import __storage__, _get_processed_datasets


class DataLoader:

    __datasets__ = _get_processed_datasets()

    def __init__(
        self,
        dataset_name: str,
        batch_size: int = 1,
        train: bool = True,
        development: bool = True,
    ):
        """Loader of sequential recommendation datasets

        Args:
            dataset_name (str): dataset name.
            batch_size (int): batch_size.
            train (bool, optional): load training data or test data. Defaults to True.
            development (bool, optional): use the dataset for hyperparameter searching. Defaults to True.
        
        Note: training data is shuffled automatically.
        """

        if dataset_name not in self.__datasets__:
            raise ValueError(
                "Dataset not found! Existing datasets: {}".format(
                    "none"
                    if len(self.__datasets__) == 0
                    else ", ".join(self.__datasets__)
                )
            )

        data_path = "./datasets/{}/processed/{}/{}.pkl".format(
            dataset_name, mode, "train" if train else "test"
        )
        with open(data_path, "rb") as f:
            self.dataset = pickle.load(f)  # list

        self.batch_size = batch_size
        self.train = train
        self._batch_count = 0

    def __iter__(self):
        return self

    def __len__(self):
        """ Number of batches """
        return math.ceil(len(self.dataset) / self.batch_size)

    def __next__(self):
        """
        Returns:
            user_ids: (batch_size,)
            input sequences: (batch_size, input_len)
            target sequences: (batch_size, target_len)
            negative samples: (batch_size, target_len, n_negatives) # Train only
        """
        if self._batch_count == len(self):
            self._batch_count = 0
            raise StopIteration
        else:
            if self._batch_count == 0 and self.train:
                random.shuffle(self.dataset)
            batch = self.dataset[
                self._batch_count
                * self.batch_size : (self._batch_count + 1)
                * self.batch_size
            ]
            self._batch_count += 1
            return list(zip(*batch))
