import pickle

import torch
import torch.utils.data

from srdatasets.datasets import __datasets__
from srdatasets.utils import __warehouse__, get_processed_datasets


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

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        users = torch.tensor(self.dataset[idx][0], dtype=torch.long)
        input_items = torch.tensor(self.dataset[idx][1], dtype=torch.long)
        target_items = torch.tensor(self.dataset[idx][2], dtype=torch.long)
        return users, input_items, target_items


class DataLoader(torch.utils.data.DataLoader):

    _processed_datasets = get_processed_datasets()

    def __init__(
        self,
        dataset_name: str,
        config_id: str,
        batch_size: int = 1,
        train: bool = True,
        development: bool = False,
        **kwargs
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

        super(DataLoader, self).__init__(
            Dataset(dataset_name, config_id, train, development),
            batch_size=batch_size,
            shuffle=train,
            **kwargs
        )
