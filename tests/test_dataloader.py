import shutil

from srdatasets.dataloader import DataLoader
from srdatasets.download import _download
from srdatasets.process import _process
from srdatasets.utils import (
    __warehouse__,
    get_downloaded_datasets,
    get_processed_datasets,
)


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

if Args.dataset in get_processed_datasets():
    shutil.rmtree(__warehouse__.joinpath(Args.dataset, "processed"))

# For short term task
short_args = Args()
_process(short_args)

# For long-short term task
long_short_args = Args()
long_short_args.task = "long-short"
long_short_args.session_interval = 60
_process(long_short_args)


def test_dataloader():
    config_ids = get_processed_datasets()[Args.dataset]
    for cid in config_ids:
        dataloader = DataLoader(
            Args.dataset,
            cid,
            batch_size=32,
            negatives_per_target=5,
            include_timestamp=True,
            drop_last=True,
        )
        if len(dataloader.dataset[0]) > 5:
            for (
                users,
                pre_sess_items,
                cur_sess_items,
                target_items,
                pre_sess_timestamps,
                cur_sess_timestamps,
                target_timestamps,
                negatives,
            ) in dataloader:
                assert users.shape == (32,)
                assert pre_sess_items.shape == (
                    32,
                    Args.pre_sessions * Args.max_session_len,
                )
                assert cur_sess_items.shape == (
                    32,
                    Args.max_session_len - Args.target_len,
                )
                assert target_items.shape == (32, Args.target_len)
                assert pre_sess_timestamps.shape == (
                    32,
                    Args.pre_sessions * Args.max_session_len,
                )
                assert cur_sess_timestamps.shape == (
                    32,
                    Args.max_session_len - Args.target_len,
                )
                assert target_timestamps.shape == (32, Args.target_len)
                assert negatives.shape == (32, Args.target_len, 5)
        else:
            for (
                users,
                in_items,
                out_items,
                in_timestamps,
                out_timestamps,
                negatives,
            ) in dataloader:
                assert users.shape == (32,)
                assert in_items.shape == (32, Args.input_len)
                assert out_items.shape == (32, Args.target_len)
                assert in_timestamps.shape == (32, Args.input_len)
                assert out_timestamps.shape == (32, Args.target_len)
                assert negatives.shape == (32, Args.target_len, 5)


# TODO Test Pytorch version DataLoader
