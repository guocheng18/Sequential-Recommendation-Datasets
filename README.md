# Sequential Recommendation Datasets
This repo simplifies how sequential recommendation datasets are used.
<p>
    <img src="https://img.shields.io/badge/python->=3.5-brightgreen?style=flat-square"/>
    <img src="https://img.shields.io/badge/pandas->=0.24-brightgreen?style=flat-square"/>
</p>

## Included datasets
- MovieLens-20M
- Last.fm-360K
- Gowalla

## Data format

## Usage

1. Download a dataset, for example `MovieLens-20M`, run
```bash
bash download_scripts/movielens-20m.sh
```
2. Generate processed dataset, run
```bash
python generate/movielens-20m.py # add -h option to see possible settings
```
3. Use DataLoader to get data batchly, for example:
```python
from dataloader import DataLoader

# For development (tune hyperparameters)
trainloader = DataLoader("movielens-20m", batch_size=32, Train=True, development=True)
testloader = DataLoader("movielens-20m", batch_size=32, Train=False, development=True)

# For performance test
trainloader = DataLoader("movielens-20m", batch_size=32, Train=True, development=False)
testloader = DataLoader("movielens-20m", batch_size=32, Train=False, development=False)

for epoch in range(10):

    # Train
    for users, inputs, targets, negatives in trainloader:
        # Shapes
        # users: (batch_size,)
        # inputs: (batch_size, input_len)
        # targets: (batch_size, target_len)
        # negatives: (batch_size, target_len, n_negatives)
        ...

    # Evaluate
    for users, inputs, targets in testloader:
        ...
```

## Disclaimers
The datasets have their own licenses, this repo (under MIT license) only provides an way to use them.