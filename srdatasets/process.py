import json
import logging
import math
import os
import pickle
import time
from pathlib import Path

from tqdm import tqdm

from srdatasets.datasets import dataset_classes
from srdatasets.utils import __warehouse__

tqdm.pandas()

logger = logging.getLogger(__name__)


def _process(args):
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
        "no_augment": args.no_augment,
    }
    if classname in ["Amazon", "MovieLens20M", "Yelp"]:
        config["rating_threshold"] = args.rating_threshold
    elif classname == "Lastfm1K":
        config["item_type"] = args.item_type

    logger.info("Transforming...")
    if classname == "Amazon":
        df = d.transform(sub, args.rating_threshold)
    elif classname in ["MovieLens20M", "Yelp"]:
        df = d.transform(args.rating_threshold)
    elif classname == "FourSquare":
        df = d.transform(sub)
    elif classname == "Lastfm1K":
        df = d.transform(args.item_type)
    else:
        df = d.transform()
    preprocess_and_save(df, args.dataset, config)


def preprocess_and_save(df, dname, config):
    """General preprocessing method
    
    Args:
        df (DataFrame): columns: `user_id`, `item_id`, `timestamp`.
        args (Namespace): arguments.
    """
    # Generate sequences
    logger.info("Generating user sequences...")
    seqs = generate_sequences(df, config["min_freq_item"], config["min_freq_user"])
    # Split sequences
    logger.info("Splitting user sequences into train/test...")
    train_seqs, test_seqs = split_sequences(
        seqs.to_dict(), config["target_len"], config["test_ratio"]
    )
    logger.info("Splitting train into dev-train/dev-test...")
    dev_train_seqs, dev_test_seqs = split_sequences(
        train_seqs, config["target_len"], config["dev_ratio"]
    )
    # Make datasets
    logger.info("Making datasets...")
    train_data, test_data, dev_train_data, dev_test_data = [
        make_dataset(
            seqs, config["input_len"], config["target_len"], config["no_augment"]
        )
        for seqs in [train_seqs, test_seqs, dev_train_seqs, dev_test_seqs]
    ]
    # Reassign user_ids and item_ids
    logger.info("Reassigning ids...")
    train_data, test_data = reassign_ids(train_data, test_data)
    dev_train_data, dev_test_data = reassign_ids(dev_train_data, dev_test_data)
    # Dump to disk
    logger.info("Dumping...")
    processed_path = __warehouse__.joinpath(
        dname, "processed", "c" + str(int(time.time() * 1000))
    )
    dump(processed_path, train_data, test_data, "test")
    dump(processed_path, dev_train_data, dev_test_data, "dev")
    with open(processed_path.joinpath("config.json"), "w") as f:
        json.dump(config, f)
    logger.info("OK")


def reassign_ids(train_data, test_data):
    train_data_ = []
    test_data_ = []
    user_to_idx = {}
    item_to_idx = {-1: 0}
    # Train collect
    for user, input_i, target_i, input_t, target_t in tqdm(train_data):
        if user not in user_to_idx:
            user_to_idx[user] = len(user_to_idx)
        user_ = user_to_idx[user]
        for i in input_i + target_i:
            if i not in item_to_idx:
                item_to_idx[i] = len(item_to_idx)
        input_i_ = [item_to_idx[i] for i in input_i]
        target_i_ = [item_to_idx[i] for i in target_i]
        train_data_.append((user_, input_i_, target_i_, input_t, target_t))
    # Test apply
    for user, input_i, target_i, input_t, target_t in tqdm(test_data):
        user_ = user_to_idx[user]
        input_i_ = [item_to_idx[i] for i in input_i]
        target_i_ = [item_to_idx[i] for i in target_i]
        test_data_.append((user_, input_i_, target_i_, input_t, target_t))
    return train_data_, test_data_


def generate_sequences(df, min_freq_item, min_freq_user):
    logger.warning("Dropping items (freq < {})...".format(min_freq_item))
    df = drop_items(df, min_freq_item)

    logger.warning("Dropping users (freq < {})...".format(min_freq_user))
    df = drop_users(df, min_freq_user)

    logger.info("Grouping items by user...")
    df = df.sort_values("timestamp")
    df["item_and_time"] = list(zip(df["item_id"], df["timestamp"]))
    seqs = (
        df.groupby("user_id")["item_and_time"]
        .progress_apply(list)
        .reset_index(drop=True)
    )
    return seqs


def split_sequences(user_seq, target_len, test_ratio):
    """Split user sequences into train/test subsequences
    """
    train_seqmap = {}
    test_seqmap = {}
    items = set()
    for user_id, seq in tqdm(user_seq.items()):
        train_len = math.floor(len(seq) * (1 - test_ratio))
        test_len = len(seq) - train_len
        # Split
        if train_len > target_len:
            if test_len > target_len:
                train_seqmap[user_id] = seq[:train_len]
                test_seqmap[user_id] = seq[train_len:]
            else:
                train_seqmap[user_id] = seq
        else:
            if len(seq) > target_len:
                train_seqmap[user_id] = seq
        # Count items
        if user_id in train_seqmap:
            for item_id, _ in train_seqmap[user_id]:
                items.add(item_id)
    # Clear new items
    for user_id, seq in list(test_seqmap.items()):
        seq_ = [(i, t) for i, t in seq if i in items]
        if len(seq_) > target_len:
            test_seqmap[user_id] = seq_
        else:
            del test_seqmap[user_id]
    return train_seqmap, test_seqmap


def make_dataset(user_seq, input_len, target_len, no_augment):
    dataset = []
    for user_id, seq in tqdm(user_seq.items()):
        if len(seq) < input_len + target_len:
            padding_num = input_len + target_len - len(seq)
            dataset.append(
                (
                    user_id,
                    [(-1, -1)] * padding_num + seq[:-target_len],
                    seq[-target_len:],
                )
            )
        elif len(seq) == input_len + target_len:
            dataset.append((user_id, seq[:-target_len], seq[-target_len:]))
        else:
            if no_augment:
                dataset.append(
                    (
                        user_id,
                        seq[-target_len - input_len : -target_len],
                        seq[-target_len:],
                    )
                )
            else:
                augmented_seqs = [
                    (
                        user_id,
                        seq[i : i + input_len],
                        seq[i + input_len : i + input_len + target_len],
                    )
                    for i in range(len(seq) - input_len - target_len + 1)
                ]
                dataset.extend(augmented_seqs)
    dataset_ = []
    for data in dataset:
        input_items, input_timestamps = list(zip(*data[1]))
        target_items, target_timestamps = list(zip(*data[2]))
        dataset_.append(
            (data[0], input_items, target_items, input_timestamps, target_timestamps)
        )
    return dataset_


def drop_users(df, min_freq):
    counts = df["user_id"].value_counts()
    df = df[df["user_id"].isin(counts[counts >= min_freq].index)]
    return df


def drop_items(df, min_freq):
    counts = df["item_id"].value_counts()
    df = df[df["item_id"].isin(counts[counts >= min_freq].index)]
    return df


def dump(path, train_data, test_data, mode):
    """ Save preprocessed datasets """
    os.makedirs(path.joinpath(mode))
    with open(path.joinpath(mode, "train.pkl"), "wb") as f:
        pickle.dump(train_data, f)
    with open(path.joinpath(mode, "test.pkl"), "wb") as f:
        pickle.dump(test_data, f)
    # write statistics
    users = set()
    items = set()
    interactions = 0
    for user, input_items, target_items, _, _ in train_data:
        users.add(user)
        for item in input_items + target_items:
            if item > 0:
                items.add(item)
                interactions += 1
    stats = {"users": len(users), "items": len(items), "interactions": interactions}
    with open(path.joinpath(mode, "stats.json"), "w") as f:
        json.dump(stats, f)


# ====== API for custom dataset ====== #
