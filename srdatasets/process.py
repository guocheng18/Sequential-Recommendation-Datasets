import json
import logging
import math
import os
import pickle
import time
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Tuple

from pandas import DataFrame, Series

from srdatasets.datasets import dataset_classes
from srdatasets.utils import __warehouse__

# TODO add negative samples, timestamps etc

logger = logging.getLogger(__name__)

Sequence = List[int]
SequenceMap = Dict[int, Sequence]
Data = Tuple[int, Sequence, Sequence]
Dataset = List[Data]


def _process(args: Namespace) -> None:
    if "-" in args.dataset:
        classname, sub = args.dataset.split("-")
    else:
        classname = args.dataset
    d = dataset_classes[classname](__warehouse__.joinpath(args.dataset, "raw"))

    config = {
        "min_freq_user": args.min_freq_user,
        "min_freq_item": args.min_freq_item,
        "dev_ratio": args.dev_ratio,
        "test_ratio": args.test_ratio,
        "input_len": args.input_len,
        "target_len": args.target_len,
    }
    
    logger.info("Transforming...")
    if classname in ["Amazon", "MovieLens-20M", "Yelp"]:
        df = d.transform(args.rating_threshold)
        config["rating_threshold"] = args.rating_threshold
    elif classname == "FourSquare":
        df = d.transform(sub)
    elif classname == "Lastfm1K":
        df = d.transform(args.item_type)
        config["item_type"] = args.item_type
    else:
        df = d.transform()
    preprocess_and_save(df, args.dataset, config)


def preprocess_and_save(df: DataFrame, dname: str, config: Dict) -> None:
    """General preprocessing method
    
    Args:
        df (DataFrame): columns: `user_id`, `item_id`, `timestamp`.
        args (Namespace): arguments.
    """
    # Generate sequences
    seqs = generate_sequences(df, config["min_freq_item"], config["min_freq_user"])
    # Create datasets
    logger.info("Splitting sequences into train/test...")
    d_train, d_test = split_sequences(
        seqs.to_dict(), config["target_len"], config["test_ratio"]
    )
    logger.info("Splitting sequences into train-train/train-dev...")
    d_train_dev, d_test_dev = split_sequences(
        d_train, config["target_len"], config["dev_ratio"]
    )
    # Augment datasets
    logger.info("Augmenting train/test/train-train/train-dev...")
    d_train, d_test, d_train_dev, d_test_dev = [
        augment_dataset(d, config["input_len"], config["target_len"])
        for d in [d_train, d_test, d_train_dev, d_test_dev]
    ]
    # Dump to files
    logger.info("Dumping...")
    processed_path = __warehouse__.joinpath(
        dname, "processed", "c" + str(int(time.time() * 1000))
    )
    dump(processed_path, d_train, d_test, "test")
    dump(processed_path, d_train_dev, d_test_dev, "dev")
    with open(processed_path.joinpath("config.json"), "w") as f:
        json.dump(config, f)
    logger.info("OK")


def generate_sequences(df: DataFrame, min_freq_item: int, min_freq_user: int) -> Series:
    """When renumbering items, 0 is kept for padding sequences
    """
    logger.warning("Dropping items (freq < {})...".format(min_freq_item))
    df = drop_infrequent_items(df, min_freq_item)

    logger.warning("Dropping users (freq < {})...".format(min_freq_user))
    df = drop_infrequent_users(df, min_freq_user)

    logger.info("Remapping item ids...")
    item_mapper = dict(
        zip(df["item_id"].unique(), range(1, df["item_id"].nunique() + 1))
    )
    df["item_id"] = df["item_id"].map(item_mapper)

    logger.info("Generating all users' sequences...")
    df = df.sort_values("timestamp")
    seqs = df.groupby("user_id")["item_id"].apply(list).reset_index(drop=True)
    return seqs


def split_sequences(
    seqs: SequenceMap, target_len: int, test_ratio: float
) -> Tuple[SequenceMap, SequenceMap]:
    """Split sequences into train/test subsequences
    """
    train_seqmap = {}
    test_seqmap = {}
    itemset = set()
    for user_id, seq in seqs.items():
        train_len = math.floor(len(seq) * (1 - test_ratio))
        if train_len > target_len:
            train_seqmap[user_id] = seq[:train_len]
            itemset.update(seq[:train_len])
        else:
            pass  # drop
        if len(seq) - train_len > target_len:
            test_seqmap[user_id] = seq[train_len:]
        else:
            pass  # drop
    for user_id, seq in list(test_seqmap.items()):
        # Filter out items that not in trainset
        seq_new = [i for i in seq if i in itemset]
        if len(seq_new) > target_len:
            test_seqmap[user_id] = seq_new
        else:
            del test_seqmap[user_id]
    return train_seqmap, test_seqmap


def augment_dataset(dataset: SequenceMap, input_len: int, target_len: int) -> Dataset:
    augmented_dataset = []
    for user_id, seq in dataset.items():
        augmented_seqs = augment_sequence(seq, user_id, input_len, target_len)
        augmented_dataset.extend(augmented_seqs)
    return augmented_dataset


def augment_sequence(
    seq: Sequence, user_id: int, input_len: int, target_len: int
) -> List[Data]:
    """ `seq` is assumed to be longer than `target_len`, 
    this has been guaranteed when splitting sequences
    """
    lack_num = input_len + target_len - len(seq)
    if lack_num > 0:
        # padding 0
        datalist = [(user_id, [0] * lack_num + seq[:-target_len], seq[-target_len:])]
    else:
        datalist = [
            (
                user_id,
                seq[i : i + input_len],
                seq[i + input_len : i + input_len + target_len],
            )
            for i in range(1 - lack_num)
        ]
    return datalist


def drop_infrequent_users(df: DataFrame, min_freq: int) -> DataFrame:
    counts = df.user_id.value_counts()
    df = df[df.user_id.isin(counts[counts.ge(min_freq)].index)]
    return df


def drop_infrequent_items(df: DataFrame, min_freq: int) -> DataFrame:
    counts = df.item_id.value_counts()
    df = df[df.item_id.isin(counts[counts.ge(min_freq)].index)]
    return df


def dump(path: Path, d_train: Dataset, d_test: Dataset, mode: str) -> None:
    """ Save preprocessed datasets """
    os.makedirs(path.joinpath(mode))
    with open(path.joinpath(mode, "train.pkl"), "wb") as f:
        pickle.dump(d_train, f)
    with open(path.joinpath(mode, "test.pkl"), "wb") as f:
        pickle.dump(d_test, f)
    # write statistics
    users = set()
    items = set()
    interactions = 0
    for user_id, inputs, targets in d_train:
        users.add(user_id)
        items.update(inputs + targets)
        interactions += len(inputs)
    stats = {"users": len(users), "items": len(items), "interactions": interactions}
    with open(path.joinpath(mode, "stats.json"), "w") as f:
        json.dump(stats, f)
