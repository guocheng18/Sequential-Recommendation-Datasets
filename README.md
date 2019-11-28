# Sequential Recommendation Datasets

Provide a tool for help dealing with some common sequential recommendation datasets

[![PyPI version](https://badge.fury.io/py/srdatasets.svg)](https://badge.fury.io/py/srdatasets)
[![Actions Status](https://github.com/guocheng2018/sequential-recommendation-datasets/workflows/Test%20and%20Publish/badge.svg)](https://github.com/guocheng2018/sequential-recommendation-datasets/actions)
[![codecov](https://codecov.io/gh/guocheng2018/sequential-recommendation-datasets/branch/master/graph/badge.svg?token=lewxFzWM0y)](https://codecov.io/gh/guocheng2018/sequential-recommendation-datasets)

## Datasets

- [Amazon-Books](http://jmcauley.ucsd.edu/data/amazon/)
- [Amazon-Electronics](http://jmcauley.ucsd.edu/data/amazon/)
- [Amazon-Movies](http://jmcauley.ucsd.edu/data/amazon/)
- [Amazon-CDs](http://jmcauley.ucsd.edu/data/amazon/)
- [Amazon-Clothing](http://jmcauley.ucsd.edu/data/amazon/)
- [Amazon-Home](http://jmcauley.ucsd.edu/data/amazon/)
- [Amazon-Kindle](http://jmcauley.ucsd.edu/data/amazon/)
- [Amazon-Sports](http://jmcauley.ucsd.edu/data/amazon/)
- [Amazon-Phones](http://jmcauley.ucsd.edu/data/amazon/)
- [Amazon-Health](http://jmcauley.ucsd.edu/data/amazon/)
- [Amazon-Toys](http://jmcauley.ucsd.edu/data/amazon/)
- [Amazon-VideoGames](http://jmcauley.ucsd.edu/data/amazon/)
- [Amazon-Tools](http://jmcauley.ucsd.edu/data/amazon/)
- [Amazon-Beauty](http://jmcauley.ucsd.edu/data/amazon/)
- [Amazon-Apps](http://jmcauley.ucsd.edu/data/amazon/)
- [Amazon-Office](http://jmcauley.ucsd.edu/data/amazon/)
- [Amazon-Pet](http://jmcauley.ucsd.edu/data/amazon/)
- [Amazon-Automotive](http://jmcauley.ucsd.edu/data/amazon/)
- [Amazon-Grocery](http://jmcauley.ucsd.edu/data/amazon/)
- [Amazon-Patio](http://jmcauley.ucsd.edu/data/amazon/)
- [Amazon-Baby](http://jmcauley.ucsd.edu/data/amazon/)
- [Amazon-Music](http://jmcauley.ucsd.edu/data/amazon/)
- [Amazon-MusicalInstruments](http://jmcauley.ucsd.edu/data/amazon/)
- [Amazon-InstantVideo](http://jmcauley.ucsd.edu/data/amazon/)
- [TaFeng](https://stackoverflow.com/a/25460645/8810037)
- [Taobao](https://tianchi.aliyun.com/dataset/dataDetail?dataId=649)
- [Tmall](https://tianchi.aliyun.com/dataset/dataDetail?dataId=47)
- [CiteULike](http://konect.cc/networks/citeulike-ut/)
- [FourSquare-NYC](https://sites.google.com/site/yangdingqi/home/foursquare-dataset)
- [FourSquare-Tokyo](https://sites.google.com/site/yangdingqi/home/foursquare-dataset)
- [Gowalla](https://snap.stanford.edu/data/loc-Gowalla.html)
- [Lastfm1K](http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html)
- [MovieLens20M](https://grouplens.org/datasets/movielens/)
- [Yelp](https://www.yelp.com/dataset)

## Install this tool

```bash
pip install -U srdatasets
```

## Download datasets

Run the command below to download datasets. Note, since some datasets are not directly accessible, you'll be warned then to download them manually and place them somewhere it tells you.

```bash
python -m srdatasets download --dataset=[dataset_name]
```

To get a view of downloaded and processed status of all datasets, run

```bash
python -m srdatasets info
```

## Process datasets

The generic processing command is

```bash
python -m srdatasets process --dataset=[dataset_name] [--options]
```

### Splitting options

Two dataset splitting methods are provided: user-based and time-based. User-based means splitting is executed on every user hehavior sequence given the ratio of validation set and test set while time-based means splitting is based on the date of user behaviors. After splitting some dataset, two processed datasets are generated, one for development, which uses the validation set as the test set, the other for test, which contains the full training set.

```code
--split-by     User or time (default: user)
--test-split   Proportion of test set to full dataset (default: 0.2)
--dev-split    Proportion of validation set to full training set (default: 0.1)
```

**NOTE**: time-based splitting need you to manually input days at console by tipping you total days of that dataset, since you may not know.

### Task related options

For **short term** recommnedation task, you use previous `input-len` items to predict next `target-len` items. To make user interests more focused, user behavior sequences can also be cut into multiple sessions if `session-interval` is given. If the number of previous items is smaller than `input-len`, 0 is padded to the left.

For **long-short term** recommendation task, you use `pre-sessions` previous sessions and current session to predict `target-len` items. The target items are picked randomly or lastly from current session. So the length of current session is `max-session-len` - `target-len` while the length of any previous session is `max-session-len`. If any previous session or current session is shorter than the preset length, 0 is padded to the left.

```code
--task              Short or long-short (default: short)
--input-len         Number of previous items (default: 5)
--target-len        Number of target items (default: 1)
--pre-sessions      Number of previous sessions (default: 10)
--pick-targets      Randomly or lastly pick items from current session (default: random)
--session-interval  Session splitting interval (minutes)  (default: 0)
--min-session-len   Sessions less than this in length will be dropped  (default: 2)
--max-session-len   Sessions greater than this in length will be cut  (default: 20)
```

### Common options

```code
--min-freq-item        Items less than this in frequency will be dropped (default: 5)
--min-freq-user        Users less than this in frequency will be dropped (default: 5)
--no-augment           Do not use data augmentation (default: False)
--remove-duplicates    Remove duplicated items in user sequence or user session (if splitted) (default: False)
```

### Dataset related options

```code
--rating-threshold  Interactions with rating less than this will be dropped (Amazon, Movielens, Yelp) (default: 4)
--item-type         Recommend artists or songs (Lastfm) (default: song)
```

### Version

By using different options, a dataset will have many processed versions. You can run the command below to get configurations and statistics of all processed versions of some dataset. The `config id` shown in output is a required argument of `DataLoader`.

```bash
python -m srdatasets info --dataset=[dataset_name]
```

## DataLoader

DataLoader is a built-in class that makes loading processed datasets easy. Practically, once initialized a dataloder by passing the dataset name, processed version (config id), batch_size and a flag to load training data or test data, you can then loop it to get batch data. Considering that some models use rank-based leanring, negative sampling is intergrated into DataLoader. The negatives are sampled from all items except items in current data according to popularity. By default it is turned off. Also, the time of user behaviors is sometimes an important feature, you can include it into batch data by setting `include_timestmap` to True.

### Arguments

- `dataset_name`: dataset name (case insensitive)
- `config_id`: configuration id
- `batch_size`: batch size (default: 1)
- `train`: load training dataset (default: True)
- `development`: load the dataset aiming for development (default: False)
- `negatives_per_target`: number of negative samples per target (default: 0)
- `include_timestamp`: add timestamps to batch data (default: False)
- `drop_last`: drop last incomplete batch (default: False)

### Initialization example

```python
from srdatasets.dataloader import DataLoader

trainloader = DataLoader("amazon-books", "c1574673118829", batch_size=32, Train=True, negatives_per_target=5, include_timestamp=True)
testloader = DataLoader("amazon-books", "c1574673118829", batch_size=32, Train=False, include_timestamp=True)
```

For pytorch users, there is a wrapper implementation of `torch.utils.data.DataLoader`, you can then set keyword arguments like `num_workers` and `pin_memory` to speed up loading data

```python
from srdatasets.dataloader_pytorch import DataLoader

trainloader = DataLoader("amazon-books", "c1574673118829", batch_size=32, Train=True, negatives_per_target=5, include_timestamp=True, num_workers=8, pin_memory=True)
testloader = DataLoader("amazon-books", "c1574673118829", batch_size=32, Train=False, include_timestamp=True, num_workers=8, pin_memory=True)
```

### Iteration template

For short term recommendation task

```python
for epoch in range(10):
    # Train
    for users, input_items, target_items, input_item_timestamps, target_item_timestamps, negative_samples in trainloader:
        # Shape
        #   users:                  (batch_size,)
        #   input_items:            (batch_size, input_len)
        #   target_items:           (batch_size, target_len)
        #   input_item_timestamps:  (batch_size, input_len)
        #   target_item_timestamps: (batch_size, target_len)
        #   negative_samples:       (batch_size, target_len, negatives_per_target)
        #
        # DataType
        #   numpy.ndarray or torch.LongTensor
        pass

    # Evaluate
    for users, input_items, target_items, input_item_timestamps, target_item_timestamps in testloader:
        pass
```

For long-short term recommendation task

```python
for epoch in range(10):
    # Train
    for users, pre_sessions_items, cur_session_items, target_items, pre_sessions_item_timestamps, cur_session_item_timestamps, target_item_timestamps, negative_samples in trainloader:
        # Shape
        #   users:                          (batch_size,)
        #   pre_sessions_items:             (batch_size, pre_sessions * max_session_len)
        #   cur_session_items:              (batch_size, max_session_len - target_len)
        #   target_items:                   (batch_size, target_len)
        #   pre_sessions_item_timestamps:   (batch_size, pre_sessions * max_session_len)
        #   cur_session_item_timestamps:    (batch_size, max_session_len - target_len)
        #   target_item_timestamps:         (batch_size, target_len)
        #   negative_samples:               (batch_size, target_len, negatives_per_target)
        #
        # DataType
        #   numpy.ndarray or torch.LongTensor
        pass

    # Evaluate
    for users, pre_sessions_items, cur_session_items, target_items, pre_sessions_item_timestamps, cur_session_item_timestamps, target_item_timestamps in testloader:
        pass
```

## Disclaimers

The datasets have their own licenses, this repo only provides a way to use them.
