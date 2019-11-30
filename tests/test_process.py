import copy

import pytest

from srdatasets.process import _process

datasets = [
    "Amazon-Apps",
    "CiteULike",
    "FourSquare-NYC",
    "Gowalla",
    "Lastfm1K",
    "MovieLens20M",
    "TaFeng",
]


class Args:
    dataset = None
    min_freq_item = 5
    min_freq_user = 5
    task = "short"  # check
    split_by = "user"  # check
    dev_split = 0.1
    test_split = 0.2
    input_len = 5
    target_len = 1
    no_augment = False  # check
    remove_duplicates = False  # check
    session_interval = 0
    max_session_len = 20
    min_session_len = 2
    pre_sessions = 10
    pick_targets = "random"
    rating_threshold = 4
    item_type = "song"


@pytest.mark.run(order=2)
@pytest.mark.parameterize("name", datasets)
def test_default_process(name):
    args = Args()
    args.dataset = name
    _process(args)


@pytest.mark.run(order=3)
def test_variant_process():
    """ Use single dataset, time-based splittings are tested currently
    """
    args = Args()
    args.dataset = "FourSquare-NYC"
    # short-term / split by user / without sessions
    args_s1 = copy.deepcopy(args)
    args_s1.task = "short"
    args_s1.split_by = "user"
    args_s1.session_interval = 0
    _process(args_s1)
    # short-term / split by user / with sessions
    args_s2 = copy.deepcopy(args)
    args_s2.task = "short"
    args_s2.split_by = "user"
    args_s2.session_interval = 20
    _process(args_s2)
    # short-term / split by user / without sessions / no_augment
    args_s3 = copy.deepcopy(args)
    args_s3.task = "short"
    args_s3.split_by = "user"
    args_s3.session_interval = 0
    args_s3.no_augment = True
    _process(args_s3)
    # short-term / split by user / without sessions / remove_duplicates
    args_s4 = copy.deepcopy(args)
    args_s4.task = "short"
    args_s4.split_by = "user"
    args_s4.session_interval = 0
    args_s4.remove_duplicates = True
    _process(args_s4)
    # long-short-term / split by user
    args_l1 = copy.deepcopy(args)
    args_l1.task = "long-short"
    args_l1.split_by = "user"
    args_l1.session_interval = 20
    _process(args_l1)
    # long-short-term / split by user / no_augment
    args_l2 = copy.deepcopy(args)
    args_l2.task = "long-short"
    args_l2.split_by = "user"
    args_l2.session_interval = 20
    args_l2.no_augment = True
    _process(args_l2)
    # long-short-term / split by user / remove_duplicates
    args_l3 = copy.deepcopy(args)
    args_l3.task = "long-short"
    args_l3.split_by = "user"
    args_l3.session_interval = 20
    args_l3.remove_duplicates = True
    _process(args_l3)
