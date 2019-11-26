# Sequential Recommendation Datasets

Provide a tool for help dealing with some common sequential recommendation datasets

[![name](https://img.shields.io/badge/pypi_package-v0.0.3-blue?style=flat-square&logo=pypi)](https://pypi.org/project/srdatasets)

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
- [CiteULike](http://konect.cc/networks/citeulike-ut/)
- [FourSquare-NYC](https://sites.google.com/site/yangdingqi/home/foursquare-dataset)
- [FourSquare-Tokyo](https://sites.google.com/site/yangdingqi/home/foursquare-dataset)
- [Gowalla](https://snap.stanford.edu/data/loc-Gowalla.html)
- [Lastfm1K](http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html)
- [MovieLens20M](https://grouplens.org/datasets/movielens/)
- [TaFeng](https://stackoverflow.com/a/25460645/8810037)
- [Taobao](https://tianchi.aliyun.com/dataset/dataDetail?dataId=649)
- [Tmall](https://tianchi.aliyun.com/dataset/dataDetail?dataId=47)
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

- --split-by: user or time (default: user)
- --dev-split: proportion of validation set to full training set (default: 0.1)
- --test-split: proportion of test set to full dataset (default: 0.2)

NOTE: time-based splitting need you to manually input days at console by tipping you total days of that dataset, since you may not know.

### Task related options

For short term recommnedation task, you use previous `input-len` items to predict next `target-len` items. To make user interests more focused, user behavior sequences can also be cut into multiple sessions if `session-interval` is given. If the number of previous items is smaller than `input-len`, 0 is padded to the left.

For long-short term recommendation task, you use `pre-sessions` previous sessions and current session to predict `target-len` items. The target items are picked randomly or lastly from current session. So the length of current session is `max-session-len` - `target-len` while the length of any previous session is `max-session-len`. If any previous session or current session is shorter than the preset length, 0 is padded to the left.

- --task: short or long-short (default: short)
- --input-len: number of previous items (for short term task) (default: 5)
- --target-len: number of target items (default: 1)
- --pre-sessions: number of previous sessions (for long-short term task) (default: 10)
- --pick-targets: randomly or lastly pick items from current session (for long-short term task) (default: random)
- --session-interval: session splitting interval (minutes)  (default: 0)
- --min-session-len: sessions less than this in length will be dropped  (default: 2)
- --max-session-len: sessions greater than this in length will be cut  (default: 20)

### Common options

- --min-freq-item: items less than this in frequency will be dropped (default: 5)
- --min-freq-user: users less than this in frequency will be dropped (default: 5)
- --no-augment: do not use data augmentation (default: False)
- --remove-duplicates: remove duplicated items in user sequence or user session (if splitted) (default: False)

### Dataset related options

- --rating-threshold: interactions with rating less than this will be dropped (Amazon, Movielens, Yelp) (default: 4)
- --item-type: recommend artists or songs (Lastfm) (default: song)

### Version

By using different options, a dataset will have many processed versions. You can run the command below to get configurations and statistics of all processed versions of some dataset

```bash
python -m srdatasets info --dataset=[dataset_name]
```

## DataLoader

DataLoader is a built-in class that helps to load the processed datasets

### Arguments

- dataset_name : dataset name (case insensitive)
- config_id : configuration id
- batch_size: batch size (default: 1)
- train: load training dataset (default: True)
- development: load the dataset aiming for development (default: False)
- negatives_per_target: number of negative samples per target (default: 0)
- include_timestamp: add timestamps to batch data (default: False)
- drop_last: drop last incomplete batch (default: False)

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

NOTE: Set `negatives_per_target` or `include_timestamp` only when your model needs

### Iteration template

For short term recommendation task

```python
for epoch in range(10):
    # Train
    for users, input_items, target_items, input_item_timestamps, target_item_timestamps, negative_samples in trainloader:
        # Shapes
        #   users: (batch_size,)
        #   input_items: (batch_size, input_len)
        #   target_items: (batch_size, target_len)
        #   input_item_timestamps: (batch_size, input_len)
        #   target_item_timestamps: (batch_size, target_len)
        #   negative_samples: (batch_size, target_len, negatives_per_target)
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
        # Shapes
        #   users: (batch_size,)
        #   pre_sessions_items: (batch_size, pre_sessions * max_session_len)
        #   cur_session_items: (batch_size, max_session_len - target_len)
        #   target_items: (batch_size, target_len)
        #   pre_sessions_item_timestamps: (batch_size, pre_sessions * max_session_len)
        #   cur_session_item_timestamps: (batch_size, max_session_len - target_len)
        #   target_item_timestamps: (batch_size, target_len)
        #   negative_samples: (batch_size, target_len, negatives_per_target)
        pass

    # Evaluate
    for users, pre_sessions_items, cur_session_items, target_items, pre_sessions_item_timestamps, cur_session_item_timestamps, target_item_timestamps in testloader:
        pass
```

NOTE: The default `DataLoader` use `numpy.Array` to represent data while the pytorch version use `torch.LongTensor`

## Disclaimers

The datasets have their own licenses, this repo only provides a way to use them.
