from srdatasets.download import _download
from srdatasets.process import _process
from srdatasets.utils import get_downloaded_datasets

# ===== Integration testing =====


class Args:
    dataset = "FourSquare-NYC"
    min_freq_item = 10
    min_freq_user = 10
    task = "short"
    split_by = "user"
    dev_split = 0.1
    test_split = 0.2
    input_len = 9
    target_len = 1
    session_interval = 0
    max_session_len = 10
    min_session_len = 2
    pre_sessions = 10
    pick_targets = "random"
    no_augment = False
    remove_duplicates = False


if Args.dataset not in get_downloaded_datasets():
    _download(Args.dataset)


def test_process_short_user():
    args = Args()
    _process(args)


def test_process_short_user_session():
    args = Args()
    args.session_interval = 60
    _process(args)


def test_process_short_time(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda: 10)
    args = Args()
    args.split_by = "time"
    _process(args)


def test_process_short_time_session(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda: 10)
    args = Args()
    args.split_by = "time"
    args.session_interval = 60
    _process(args)


def test_process_long_short_user():
    args = Args()
    args.session_interval = 60
    args.task = "long-short"
    _process(args)


def test_process_long_short_time(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda: 10)
    args = Args()
    args.split_by = "time"
    args.session_interval = 60
    args.task = "long-short"
    _process(args)


# ===== TODO Unit testing =====
