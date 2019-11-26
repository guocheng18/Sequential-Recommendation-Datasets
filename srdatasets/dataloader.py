import copy
import logging
import math
import os
import pickle
import random
from collections import Counter

import numpy as np

from srdatasets.datasets import __datasets__
from srdatasets.utils import (__warehouse__, get_datasetname,
                              get_processed_datasets)

logger = logging.getLogger(__name__)


class DataLoader:

    _processed_datasets = get_processed_datasets()

    def __init__(
        self,
        dataset_name: str,
        config_id: str,
        batch_size: int = 1,
        train: bool = True,
        development: bool = False,
        negatives_per_target: int = 0,
        include_timestamp: bool = False,
        drop_last: bool = False,
    ):
        """Loader of sequential recommendation datasets

        Args:
            dataset_name (str): dataset name.
            config_id (str): dataset config id
            batch_size (int): batch_size
            train (bool, optional): load training data
            development (bool, optional): use the dataset for hyperparameter tuning
            negatives_per_target (int, optional): number of negative samples per target
            include_timestamp (bool, optional): add timestamps to batch data
            drop_last (bool, optional): drop last incomplete batch
        
        Note: training data is shuffled automatically.
        """
        dataset_name = get_datasetname(dataset_name)

        if dataset_name not in __datasets__:
            raise ValueError(
                "Unrecognized dataset, currently supported datasets: {}".format(
                    ", ".join(__datasets__)
                )
            )

        if dataset_name not in self._processed_datasets:
            raise ValueError(
                "{} has not been processed, currently processed datasets: {}".format(
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

        if negatives_per_target < 0:
            negatives_per_target = 0
            logger.warning(
                "Number of negative samples per target should >= 0, reset to 0"
            )

        if not train and negatives_per_target > 0:
            logger.warning(
                "Negative samples are used for training, set negatives_per_target has no effect when testing"
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

        if train:
            counter = Counter()
            for data in self.dataset:
                if len(data) > 5:
                    counter.update(data[1] + data[2] + data[3])
                else:
                    counter.update(data[1] + data[2])
            self.item_counts = np.array(
                [counter[i] for i in range(max(counter.keys()) + 1)]
            )

        if batch_size <= 0:
            raise ValueError("batch_size should >= 1")
        if batch_size > len(self.dataset):
            raise ValueError("batch_size exceeds the dataset size")

        self.batch_size = batch_size
        self.train = train
        self.include_timestamp = include_timestamp
        self.negatives_per_target = negatives_per_target
        self.drop_last = drop_last
        self._batch_idx = 0

    def __iter__(self):
        return self

    def __len__(self):
        """Number of batches
        """
        if self.drop_last:
            return math.floor(len(self.dataset) / self.batch_size)
        else:
            return math.ceil(len(self.dataset) / self.batch_size)

    def sample_negatives(self, batch_items_list):
        negatives = []
        for b in np.concatenate(batch_items_list, 1):
            item_counts = copy.deepcopy(self.item_counts)
            item_counts[b] = 0
            item_counts[0] = 0
            probs = item_counts / item_counts.sum()
            _negatives = np.random.choice(
                len(item_counts),
                size=self.negatives_per_target * batch_items_list[-1].shape[1],
                replace=False,
                p=probs,
            )
            _negatives = _negatives.reshape((-1, self.negatives_per_target))
            negatives.append(_negatives)
        return np.stack(negatives)

    def __next__(self):
        """
        Returns:
            user_ids: (batch_size,)
            input sequences: (batch_size, input_len)
            target sequences: (batch_size, target_len)
        """
        if self._batch_idx == len(self):
            self._batch_idx = 0
            raise StopIteration
        else:
            if self._batch_idx == 0 and self.train:
                random.shuffle(self.dataset)
            batch = self.dataset[
                self._batch_idx
                * self.batch_size : (self._batch_idx + 1)
                * self.batch_size
            ]
            self._batch_idx += 1
            batch_data = [np.array(b) for b in zip(*batch)]
            # Diff task
            target_idx = 3 if len(batch_data) > 5 else 2
            if not self.include_timestamp:
                batch_data = batch_data[: target_idx + 1]
            # Sampling negatives
            if self.train and self.negatives_per_target > 0:
                negatives = self.sample_negatives(batch_data[1 : target_idx + 1])
                batch_data.append(negatives)
            return batch_data
