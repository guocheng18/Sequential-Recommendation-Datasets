import logging
import pickle
from collections import Counter

import torch
import torch.utils.data

from srdatasets.datasets import __datasets__
from srdatasets.utils import (__warehouse__, get_datasetname,
                              get_processed_datasets)

logger = logging.getLogger(__name__)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, name: str, config_id: str, train: bool, development: bool):
        super(Dataset, self).__init__()
        datapath = __warehouse__.joinpath(
            name,
            "processed",
            config_id,
            "dev" if development else "test",
            "train.pkl" if train else "test.pkl",
        )
        if datapath.exists():
            with open(datapath, "rb") as f:
                self.dataset = pickle.load(f)
        else:
            raise ValueError("{} does not exist!".format(datapath))
        if train:
            self.item_counts = Counter()
            for data in self.dataset:
                if len(data) > 5:
                    self.item_counts.update(data[1] + data[2] + data[3])
                else:
                    self.item_counts.update(data[1] + data[2])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return tuple(map(torch.tensor, self.dataset[idx]))


class DataLoader(torch.utils.data.DataLoader):

    _processed_datasets = get_processed_datasets()

    def collate_fn(self, batch):
        """ Negative sampling and Timestamps removal or adding
        """
        target_pos = 3 if len(batch[0]) > 5 else 2

        batch_data = list(zip(*batch))
        if self.include_timestamp:
            batch_data = list(map(torch.stack, batch_data))
        else:
            batch_data = list(map(torch.stack, batch_data[: target_pos + 1]))

        if self.train and self.negatives_per_target > 0:
            batch_item_counts = self.item_counts.repeat(len(batch), 1).scatter(
                1, torch.cat(batch_data[1 : target_pos + 1], 1), 0
            )
            # Prevent padding item 0 from being negative samples
            batch_item_counts[:, 0] = 0
            negatives = torch.multinomial(
                batch_item_counts,
                self.negatives_per_target * batch_data[target_pos].size(1),
            )
            negatives = negatives.view(len(batch), -1, self.negatives_per_target)
            batch_data.append(negatives)
        return batch_data

    def __init__(
        self,
        dataset_name: str,
        config_id: str,
        batch_size: int = 1,
        train: bool = True,
        development: bool = False,
        negatives_per_target: int = 0,
        include_timestamp: bool = False,
        **kwargs,
    ):
        """Loader of sequential recommendation datasets

        Args:
            dataset_name (str): dataset name
            config_id (str): dataset config id
            batch_size (int): batch_size
            train (bool, optional): load training dataset
            development (bool, optional): use the dataset for hyperparameter tuning
            negatives_per_target (int, optional): number of negative samples per target
            include_timestamp (bool, optional): add timestamps to batch data
        
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

        self.train = train
        self.include_timestamp = include_timestamp
        self.negatives_per_target = negatives_per_target

        _dataset = Dataset(dataset_name, config_id, train, development)
        if train:
            self.item_counts = torch.tensor(
                [
                    _dataset.item_counts[i]
                    for i in range(max(_dataset.item_counts.keys()) + 1)
                ],
                dtype=torch.float,
            )

        super(DataLoader, self).__init__(
            _dataset,
            batch_size=batch_size,
            shuffle=train,
            collate_fn=self.collate_fn,
            **kwargs,
        )
