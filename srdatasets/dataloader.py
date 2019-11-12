import math
import os
import pickle
import random

import numpy as np

from srdatasets.datasets import __datasets__
from srdatasets.utils import __warehouse__, get_processed_datasets


class DataLoader:

    _processed_datasets = get_processed_datasets()

    def __init__(
        self,
        dataset_name: str,
        config_id: str,
        batch_size: int = 1,
        train: bool = True,
        development: bool = False,
    ):
        """Loader of sequential recommendation datasets

        Args:
            dataset_name (str): dataset name.
            config_id (str): dataset config id
            batch_size (int): batch_size.
            train (bool, optional): load training data or test data. Defaults to True.
            development (bool, optional): use the dataset for hyperparameter searching. Defaults to True.
        
        Note: training data is shuffled automatically.
        """
        if dataset_name not in __datasets__:
            raise ValueError(
                "{} is not supported, currently supported datasets: {}".format(
                    dataset_name, ", ".join(__datasets__)
                )
            )

        if dataset_name not in self._processed_datasets:
            raise ValueError(
                "{} is not processed, currently processed datasets: {}".format(
                    dataset_name,
                    ", ".join(self._processed_datasets)
                    if self._processed_datasets
                    else "none",
                )
            )

        if config_id not in self._processed_datasets[dataset_name]:
            raise ValueError(
                "Unrecognized config id, existing config ids: {}".format(
                    ", ".join(self._processed_datasets[dataset_name])
                )
            )

        dataset_path = __warehouse__.joinpath(
            dataset_name,
            "processed",
            config_id,
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
            return [np.array(b) for b in zip(*batch)]
