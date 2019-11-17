import json
import logging
import math
import os
import pickle
import sys
import time
from datetime import datetime
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
        "input_len": args.input_len,
        "target_len": args.target_len,
        "no_augment": args.no_augment,
        "remove_duplicates": args.remove_duplicates,
        "session_interval": args.session_interval,
        "split_by": args.split_by,
        "dev_split": args.dev_split,
        "test_split": args.test_split,
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

    if args.split_by == "time":
        config["dev_split"], config["test_split"] = access_split_days(df)
        # Processed check
        if (
            "time_splits" in args
            and (config["dev_split"], config["test_split"]) in args.time_splits
        ):
            logger.warning(
                "You have run this config, the config id is {}".format(
                    args.time_splits[(config["dev_split"], config["test_split"])]
                )
            )
            sys.exit(1)
        config["max_timestamp"] = df["timestamp"].max()

    preprocess_and_save(df, args.dataset, config)


def access_split_days(df):
    min_timestamp = df["timestamp"].min()
    max_timestamp = df["timestamp"].max()
    first_day = datetime.fromtimestamp(min_timestamp).strftime("%Y-%m-%d")
    last_day = datetime.fromtimestamp(max_timestamp).strftime("%Y-%m-%d")
    total_days = math.ceil((max_timestamp - min_timestamp) / 86400)
    print("Date range: {} ~ {}, total days: {}".format(first_day, last_day, total_days))
    while True:
        try:
            test_last_days = int(input("Last N days for test: "))
            dev_last_days = int(input("Last N days for dev: "))
            if test_last_days <= 0 or dev_last_days <= 0:
                raise ValueError
            elif test_last_days + dev_last_days >= total_days:
                raise AssertionError
            else:
                break
        except ValueError:
            print("Please input a positive integer!")
        except AssertionError:
            print("test_last_days + dev_last_days < total_days")
    return dev_last_days, test_last_days


def preprocess_and_save(df, dname, config):
    """General preprocessing method
    
    Args:
        df (DataFrame): columns: `user_id`, `item_id`, `timestamp`.
        args (Namespace): arguments.
    """
    # Generate sequences
    logger.info("Generating user sequences...")
    seqs = generate_sequences(df, config)

    # Split sequences in different ways
    if config["session_interval"] > 0:
        split = split_sequences_session
    else:
        split = split_sequences

    logger.info("Splitting user sequences into train/test...")
    train_seqs, test_seqs = split(seqs, config, 0)
    logger.info("Removing new items in test...")
    test_seqs = remove_new_items(train_seqs, test_seqs, config)

    logger.info("Splitting train into dev-train/dev-test...")
    dev_train_seqs, dev_test_seqs = split(train_seqs, config, 1)
    logger.info("Removing new items in dev-test...")
    dev_test_seqs = remove_new_items(dev_train_seqs, dev_test_seqs, config)

    # Remove duplicates (optional)
    if config["remove_duplicates"]:
        logger.info("Removing duplicates...")
        train_seqs, test_seqs, dev_train_seqs, dev_test_seqs = [
            remove_duplicates(seqs, config)
            for seqs in [train_seqs, test_seqs, dev_train_seqs, dev_test_seqs]
        ]

    # Make datasets
    logger.info("Making datasets...")
    train_data, test_data, dev_train_data, dev_test_data = [
        make_dataset(seqs, config)
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
    dump(processed_path, train_data, test_data, 0)
    dump(processed_path, dev_train_data, dev_test_data, 1)
    # Save config
    save_config(processed_path, config)
    logger.info("OK")


def reassign_ids(train_data, test_data):
    """ No new items or users in test_data
    """
    user_to_idx = {}
    item_to_idx = {-1: 0}
    train_data_ = []
    test_data_ = []
    for user, input_items, target_items, input_times, target_times in tqdm(train_data):
        # build user dict
        if user not in user_to_idx:
            user_to_idx[user] = len(user_to_idx)
        user_ = user_to_idx[user]
        # build item dict
        for item in input_items + target_items:
            if item not in item_to_idx:
                item_to_idx[item] = len(item_to_idx)
        input_items_ = [item_to_idx[i] for i in input_items]
        target_items_ = [item_to_idx[i] for i in target_items]
        train_data_.append(
            (user_, input_items_, target_items_, input_times, target_times)
        )
    for user, input_items, target_items, input_times, target_times in tqdm(test_data):
        user_ = user_to_idx[user]
        input_items_ = [item_to_idx[i] for i in input_items]
        target_items_ = [item_to_idx[i] for i in target_items]
        test_data_.append(
            (user_, input_items_, target_items_, input_times, target_times)
        )
    return train_data_, test_data_


def generate_sequences(df, config):
    logger.warning("Dropping items (freq < {})...".format(config["min_freq_item"]))
    df = drop_items(df, config["min_freq_item"])

    logger.warning("Dropping users (freq < {})...".format(config["min_freq_user"]))
    df = drop_users(df, config["min_freq_user"])

    logger.info("Grouping items by user...")
    df = df.sort_values("timestamp", ascending=True)
    df["item_and_time"] = list(zip(df["item_id"], df["timestamp"]))
    seqs = df.groupby("user_id")["item_and_time"].progress_apply(list)
    seqs = list(zip(seqs.index, seqs))  # no check on sequence length

    if config["session_interval"] > 0:
        logger.info("Splitting sessions...")
        _seqs = []
        for user_id, seq in tqdm(seqs):
            seq_buffer = []
            for i, (item_id, timestamp) in enumerate(seq):
                if i == 0:
                    seq_buffer.append((item_id, timestamp))
                else:
                    if timestamp - seq[i - 1][1] > config["session_interval"] * 60:
                        if len(seq_buffer) > config["target_len"]:
                            _seqs.append((user_id, seq_buffer))
                        seq_buffer = [(item_id, timestamp)]
                    else:
                        seq_buffer.append((item_id, timestamp))
            if len(seq_buffer) > config["target_len"]:
                _seqs.append((user_id, seq_buffer))
        seqs = _seqs
    return seqs


def split_sequences(user_seq, config, mode):
    """ Without sessions 
    """
    if config["split_by"] == "user":
        test_ratio = config["dev_split"] if mode else config["test_split"]
    else:
        last_days = (
            config["dev_split"] + config["test_split"] if mode else config["test_split"]
        )
        split_timestamp = config["max_timestamp"] - last_days * 86400
    train_seqs = []
    test_seqs = []
    for user_id, seq in tqdm(user_seq):
        train_num = 0
        if config["split_by"] == "user":
            train_num = math.ceil(len(seq) * (1 - test_ratio))
        else:
            for item, timestamp in seq:
                if timestamp < split_timestamp:
                    train_num += 1
        if train_num > config["target_len"]:
            train_seqs.append((user_id, seq[:train_num]))
            if len(seq) - train_num > config["target_len"]:
                test_seqs.append((user_id, seq[train_num:]))
    return train_seqs, test_seqs


def split_sequences_session(user_seq, config, mode):
    """ With sessions, when number of sessions is small, len of test_seqs can be 0
    """
    if config["split_by"] == "user":
        test_ratio = config["dev_split"] if mode else config["test_split"]
    else:
        last_days = (
            config["dev_split"] + config["test_split"] if mode else config["test_split"]
        )
        split_timestamp = config["max_timestamp"] - last_days * 86400
    train_seqs = []
    test_seqs = []
    user_sessions = []
    for i, (user_id, seq) in tqdm(enumerate(user_seq), total=len(user_seq)):
        if i == 0:
            user_sessions.append((user_id, seq))
        else:
            if user_id == user_seq[i - 1][0]:
                user_sessions.append((user_id, seq))
            else:
                if config["split_by"] == "user":
                    train_num = math.ceil((1 - test_ratio) * len(user_sessions))
                else:
                    train_num = 0
                    for session in user_sessions:
                        if session[1][0][1] < split_timestamp:
                            train_num += 1
                if train_num > 0:
                    train_seqs.extend(user_sessions[:train_num])
                    test_seqs.extend(user_sessions[train_num:])
                user_sessions = [(user_id, seq)]
    if config["split_by"] == "user":
        train_num = math.ceil((1 - test_ratio) * len(user_sessions))
    else:
        train_num = 0
        for session in user_sessions:
            if session[1][0][1] < split_timestamp:
                train_num += 1
    if train_num > 0:
        train_seqs.extend(user_sessions[:train_num])
        test_seqs.extend(user_sessions[train_num:])
    return train_seqs, test_seqs


def remove_new_items(train_seqs, test_seqs, config):
    items = set()
    for user_id, seq in tqdm(train_seqs):
        items.update([i for i, t in seq])
    test_seqs_ = []
    for user_id, seq in tqdm(test_seqs):
        seq_ = [(i, t) for i, t in seq if i in items]
        if len(seq_) > config["target_len"]:
            test_seqs_.append((user_id, seq_))
    return test_seqs_


def remove_duplicates(user_seq, config):
    """ By default, we keep the first
    """
    user_seq_ = []
    for user_id, seq in tqdm(user_seq):
        seq_ = []
        shown_items = set()
        for item, timestamp in seq:
            if item not in shown_items:
                shown_items.add(item)
                seq_.append((item, timestamp))
        if len(seq_) > config["target_len"]:
            user_seq_.append((user_id, seq_))
    return user_seq_


def make_dataset(user_seq, config):
    input_len = config["input_len"]
    target_len = config["target_len"]
    dataset = []
    for user_id, seq in tqdm(user_seq):
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
            if config["no_augment"]:
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


def cal_stats(train_data, test_data):
    users = set()
    items = set()
    interactions = 0
    for user, input_items, target_items, _, _ in train_data:
        users.add(user)
        for item in input_items + target_items:
            if item > 0:  # reassigned
                items.add(item)
                interactions += 1
    stats = {
        "users": len(users),
        "items": len(items),
        "interactions": interactions,
        "density": interactions / len(users) / len(items),
        "train size": len(train_data),
        "test size": len(test_data),
    }
    return stats


def drop_users(df, min_freq):
    counts = df["user_id"].value_counts()
    df = df[df["user_id"].isin(counts[counts >= min_freq].index)]
    return df


def drop_items(df, min_freq):
    counts = df["item_id"].value_counts()
    df = df[df["item_id"].isin(counts[counts >= min_freq].index)]
    return df


def save_config(path, config):
    if "max_timestamp" in config:
        del config["max_timestamp"]
    with open(path.joinpath("config.json"), "w") as f:
        json.dump(config, f)


def dump(path, train_data, test_data, mode):
    """ Save preprocessed datasets """
    dirname = "dev" if mode else "test"
    os.makedirs(path.joinpath(dirname))
    with open(path.joinpath(dirname, "train.pkl"), "wb") as f:
        pickle.dump(train_data, f)
    with open(path.joinpath(dirname, "test.pkl"), "wb") as f:
        pickle.dump(test_data, f)
    stats = cal_stats(train_data, test_data)
    with open(path.joinpath(dirname, "stats.json"), "w") as f:
        json.dump(stats, f)


# ====== API for custom dataset ====== #
