# Sequential Recommendation Datasets
This repo simplifies how sequential recommendation datasets are used.
<p>
    <img src="https://img.shields.io/badge/pypi package-v0.0.1-brightgreen?style=flat-square"/>
</p>

## Datasets
Name | Item | Website
---- | ---- | -------
Amazon-[category] | Product| http://jmcauley.ucsd.edu/data/amazon/
CiteULike | Tag | http://konect.cc/networks/citeulike-ut/
FourSquare-[city] | Location| https://sites.google.com/site/yangdingqi/home/foursquare-dataset
Gowalla | Location | https://snap.stanford.edu/data/loc-Gowalla.html
Lastfm1K | Artist or Music | http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html
MovieLens20M | Movie | https://grouplens.org/datasets/movielens/
TaFeng | Product | https://stackoverflow.com/questions/25014904/download-link-%20for-ta-feng-grocery-dataset
Taobao | Product | https://tianchi.aliyun.com/dataset/dataDetail?dataId=649
Tmall | Product | https://tianchi.aliyun.com/dataset/dataDetail?dataId=47
Yelp | Business | https://www.yelp.com/dataset

### NOTE
- Amazon categories: Books, Electronics, Movies, CDs, Clothing, Home, Kindle, Sports, Phones, Health, Toys, VideoGames, Tools, Beauty, Apps, Office, Pet, Automotive, Grocery, Patio, Baby, Music, MusicalInstruments, InstantVideo
- FourSquare cities: NYC, Tokyo

## Installation
Install from pypi:
```
pip install srdatasets
```
Or from Github for the latest version:
```
pip install git+https://github.com/guocheng2018/sequential-recommendation-datasets.git
```

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

## TODO
- [ ] More datasets
- [ ] Support Custom datasets


## Disclaimers
The datasets have their own licenses, this repo (under MIT license) only provides an way to use them.