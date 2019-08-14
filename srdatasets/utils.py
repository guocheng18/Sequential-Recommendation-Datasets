import os

__storage__ = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".warehouse")

__datasets__ = ["movielens-20m", "lastfm-360k", "gowalla"]


def _get_processed_datasets():
    P = [
        "processed/dev/train.pkl",
        "processed/dev/test.pkl",
        "processed/test/train.pkl",
        "processed/test/test.pkl",
    ]
    D = []
    for d in os.listdir(__storage__):
        if all([os.path.exists(os.path.join(__storage__, d, p)) for p in P]):
            D.append(d)
    return D


def _get_downloaded_datasets():  # Simple check, need improvments
    M = {
        "movielens-20m": "ratings.csv",
        "lastfm-360k": "usersha1-artmbid-artname-plays.tsv",
        "gowalla": "loc-gowalla_totalCheckins.txt",
    }
    D = []
    for d, filename in M.items():
        if os.path.exists(os.path.join(__storage__, d, "raw", filename)):
            D.append(d)
    return D
