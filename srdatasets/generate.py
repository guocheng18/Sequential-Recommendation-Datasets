import math
import os
import pickle
from argparse import Namespace
from typing import Dict, List, Optional, Tuple

from pandas import DataFrame, Series

from srdatasets.datasets import _dataset_classes
from srdatasets.utils import __warehouse__

# Typing aliases
Sequence = List[int]
SequenceMap = Dict[int, Sequence]
Data = Tuple[int, Sequence, Sequence]
Dataset = List[Data]


def _generate(args: Namespace) -> None:
    d = _dataset_classes[args.dataset](__warehouse__.joinpath(args.dataset, "raw"))
    if args.dataset == "movielens-20m":
        df = d.transform(args.rating_threshold)
    else:
        df = d.transform()
    _preprocess_and_save(df, args)


def _preprocess_and_save(df: DataFrame, args: Namespace) -> None:
    """General preprocessing method
    
    Args:
        df (DataFrame): columns: `user_id`, `item_id`, `timestamp`.
        args (Namespace): arguments.
    """
    # Generate sequences
    seqs = _generate_sequences(df, args.min_freq_item, args.min_freq_user)
    if args.verbose:
        log_sequences_info(seqs)
    # Create datasets
    d_train, d_test = _split_sequences(seqs.to_dict(), args.target_len, args.test_ratio)
    d_train_dev, d_test_dev = _split_sequences(d_train, args.target_len, args.dev_ratio)
    # Augment datasets
    d_train, d_test, d_train_dev, d_test_dev = [
        _augment_dataset(d, args.input_len, args.target_len)
        for d in [d_train, d_test, d_train_dev, d_test_dev]
    ]
    if args.verbose:
        log_datasets_info(d_train, d_test, print_title="test")
        log_datasets_info(d_train_dev, d_test_dev, print_title="dev")
    # Dump to files
    dump(d_train, d_test, args.dataset, dev=False)
    dump(d_train_dev, d_test_dev, args.dataset, dev=True)


def _generate_sequences(
    df: DataFrame, min_freq_item: int, min_freq_user: int
) -> Series:
    """When renumbering items, 0 is kept for padding sequences
    """
    df = _drop_infrequent_items(df, min_freq_item)
    df = _drop_infrequent_users(df, min_freq_user)
    item_mapper = dict(zip(df.item_id.unique(), range(1, df.item_id.nunique() + 1)))
    df.item_id = df.item_id.map(item_mapper)
    df = df.sort_values("timestamp")
    seqs = df.groupby("user_id").item_id.apply(list).reset_index(drop=True)
    return seqs


def _split_sequences(
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
        seq_new = [i for i in seq if i in itemset]
        if len(seq_new) > target_len:
            test_seqmap[user_id] = seq_new
        else:
            del test_seqmap[user_id]
    return train_seqmap, test_seqmap


def _augment_dataset(dataset: SequenceMap, input_len: int, target_len: int) -> Dataset:
    augmented_dataset = []
    for user_id, seq in dataset.items():
        augmented_seqs = _augment_sequence(seq, user_id, input_len, target_len)
        augmented_dataset.extend(augmented_seqs)
    return augmented_dataset


def _augment_sequence(
    seq: Sequence, user_id: int, input_len: int, target_len: int
) -> List[Data]:
    """ `seq` is assumed to be longer than `target_len` """
    lack_num = input_len + target_len - len(seq)
    if lack_num > 0:
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


def _drop_infrequent_users(df: DataFrame, min_freq: int) -> DataFrame:
    counts = df.user_id.value_counts()
    df = df[df.user_id.isin(counts[counts.ge(min_freq)].index)]
    return df


def _drop_infrequent_items(df: DataFrame, min_freq: int) -> DataFrame:
    counts = df.item_id.value_counts()
    df = df[df.item_id.isin(counts[counts.ge(min_freq)].index)]
    return df


def dump(d_train: Dataset, d_test: Dataset, dname: str, dev: float = False) -> None:
    """ Save preprocessed datasets """
    dump_path = os.path.join(
        __warehouse__, dname, "processed", "dev" if dev else "test"
    )
    os.makedirs(dump_path, exist_ok=True)
    with open(os.path.join(dump_path, "train.pkl"), "wb") as f:
        pickle.dump(d_train, f)
    with open(os.path.join(dump_path, "test.pkl"), "wb") as f:
        pickle.dump(d_test, f)


def log_sequences_info(seqs: Series) -> None:
    lens = seqs.apply(len)
    print(
        "{}> sequences info\ncount\tmin\tmax\tmean\n{}\t{}\t{}\t{}".format(
            "=" * 10, seqs.count(), lens.min(), lens.max(), lens.mean()
        )
    )


def log_datasets_info(
    d_train: Dataset, d_test: Dataset, print_title: Optional[str] = None
) -> None:
    users = set()
    items = set()
    for user_id, inputs, targets in d_train:
        users.add(user_id)
        items.update(inputs)
        items.update(targets)
    print(
        "{}> {}\ntotal_users\ttotal_items\ttrain_size\ttest_size\n{}\t{}\t{}\t{}".format(
            "=" * 10, print_title, len(users), len(items), len(d_train), len(d_test)
        )
    )
