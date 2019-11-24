# Sequential Recommendation Datasets

Provide a tool for help dealing with some common sequential recommendation datasets

[![name](https://img.shields.io/badge/pypi_package-v0.0.1-blue?style=flat-square&logo=pypi)](https://pypi.org/)

## Datasets

Name | ItemType | Website
---- | -------- | -------
Amazon | Product| http://jmcauley.ucsd.edu/data/amazon/
CiteULike | Tag | http://konect.cc/networks/citeulike-ut/
FourSquare | Location | https://sites.google.com/site/yangdingqi/home/foursquare-dataset
Gowalla | Location | https://snap.stanford.edu/data/loc-Gowalla.html
Lastfm1K | Artist or Music | http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html
MovieLens20M | Movie | https://grouplens.org/datasets/movielens/
TaFeng | Product | https://stackoverflow.com/a/25460645/8810037
Taobao | Product | https://tianchi.aliyun.com/dataset/dataDetail?dataId=649
Tmall | Product | https://tianchi.aliyun.com/dataset/dataDetail?dataId=47
Yelp | Business | https://www.yelp.com/dataset

## Install


## Download datasets


## Process datasets


## Usage

1. Download a dataset, for example `MovieLens-20M`
```bash
python -m srdatasets download --dataset="MovieLens-20M"
```
2. Process the downloaded dataset with details logged to console
```bash
python -m srdatasets process --dataset="MovieLens-20M"

# Add -h option to see all specific settings of dataset processing
python -m srdatasets process -h
```
3. Check local datasets info
```
python -m srdatasets info
```
4. Use `srdatasets.DataLoader` to get data batchly
```python
from srdatasets import DataLoader

# For development (tune hyperparameters)
trainloader = DataLoader("MovieLens-20M", batch_size=32, Train=True, development=True)
testloader = DataLoader("MovieLens-20M", batch_size=32, Train=False, development=True)

# For performance test
trainloader = DataLoader("MovieLens-20M", batch_size=32, Train=True, development=False)
testloader = DataLoader("MovieLens-20M", batch_size=32, Train=False, development=False)

for epoch in range(10):

    # Train
    for user_ids, input_item_ids, target_items_id in trainloader:
        # Shapes
        # user_ids: (batch_size,)
        # input_item_ids: (batch_size, input_len)
        # target_item_ids: (batch_size, target_len)
        ...

    # Evaluate
    for user_ids, input_item_ids, target_item_ids in testloader:
        ...
```

## Disclaimers
The datasets have their own licenses, this repo (under MIT license) only provides an way to use them.