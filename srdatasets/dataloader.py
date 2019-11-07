import math
import os
import pickle
import random

from srdatasets.utils import __warehouse__, _get_processed_datasets


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

        dataset_path = __warehouse__.joinpath(
            dataset_name,
            "processed",
            "dev" if development else "test",
            "train.pkl" if train else "test.pkl",
        )
        with open(dataset_path, "rb") as f:
            self.dataset = pickle.load(f)  # list

        if batch_size <= 0:
            raise ValueError("batch_size should be at least 1")
        if batch_size > len(self.dataset):
            raise ValueError("batch_size exceeds the dataset size")

        self.batch_size = batch_size
        self.train = train
        self._batch_index = 0

    def __iter__(self):
        return self

    def __len__(self):
        """Number of batches
        """
        return math.ceil(len(self.dataset) / self.batch_size)

    def __next__(self):
        """
        Returns:
            user_ids: (batch_size,)
            input sequences: (batch_size, input_len)
            target sequences: (batch_size, target_len)
        """
        if self._batch_index == len(self):
            self._batch_index = 0
            raise StopIteration
        else:
            if self._batch_index == 0 and self.train:
                random.shuffle(self.dataset)
            batch = self.dataset[
                self._batch_index
                * self.batch_size : (self._batch_index + 1)
                * self.batch_size
            ]
            self._batch_index += 1
            return list(zip(*batch))
