import copy
from argparse import Namespace

from srdatasets.download import _download
from srdatasets.process import _process
from srdatasets.utils import get_downloaded_datasets

# ===== Integration testing =====


args = Namespace(
    dataset="FourSquare-NYC",
    min_freq_item=10,
    min_freq_user=10,
    task="short",
    split_by="user",
    dev_split=0.1,
    test_split=0.2,
    input_len=9,
    target_len=1,
    session_interval=0,
    max_session_len=10,
    min_session_len=2,
    pre_sessions=10,
    pick_targets="random",
    no_augment=False,
    remove_duplicates=False,
)


if args.dataset not in get_downloaded_datasets():
    _download(args.dataset)


def test_process_short_user():
    local_args = copy.deepcopy(args)
    _process(local_args)


def test_process_short_user_session():
    local_args = copy.deepcopy(args)
    local_args.session_interval = 60
    _process(local_args)


def test_process_short_time(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt="": 10)
    local_args = copy.deepcopy(args)
    local_args.split_by = "time"
    _process(local_args)


def test_process_short_time_session(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt="": 10)
    local_args = copy.deepcopy(args)
    local_args.split_by = "time"
    local_args.session_interval = 60
    _process(local_args)


def test_process_long_short_user():
    local_args = copy.deepcopy(args)
    local_args.session_interval = 60
    local_args.task = "long-short"
    _process(local_args)


def test_process_long_short_time(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt="": 10)
    local_args = copy.deepcopy(args)
    local_args.split_by = "time"
    local_args.session_interval = 60
    local_args.task = "long-short"
    _process(local_args)


def test_no_augment_and_remove_duplicates():
    local_args = copy.deepcopy(args)
    local_args.no_augment = True
    local_args.remove_duplicates = True
    _process(local_args)

# ===== TODO Unit testing =====
