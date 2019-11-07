import argparse
import logging

from srdatasets.datasets import __datasets__
from srdatasets.download import _download
from srdatasets.generate import _generate
from srdatasets.utils import _get_downloaded_datasets, _get_processed_datasets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d] %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)

parser = argparse.ArgumentParser("python -m srdatasets")
parser.add_argument(
    "--dataset", type=str, default=None, required=True, help="dataset name"
)
subparsers = parser.add_subparsers()
parser_d = subparsers.add_parser("download", help="download raw datasets")

parser_g = subparsers.add_parser("generate", help="generate preprocessed datasets")
group_c = parser_g.add_argument_group("common arguments")
group_c.add_argument(
    "--dev-ratio", type=float, default=0.1, help="the fraction of developemnt dataset"
)
group_c.add_argument(
    "--test-ratio", type=float, default=0.2, help="the fraction of test dataset"
)
group_c.add_argument(
    "--min-freq-item", type=int, default=5, help="minimum occurrence times of item"
)
group_c.add_argument(
    "--min-freq-user", type=int, default=10, help="minimum occurrence times of user"
)
group_c.add_argument("--input-len", type=int, default=5, help="input sequence length")
group_c.add_argument("--target-len", type=int, default=3, help="target sequence length")
group_c.add_argument("--logstat", action="store_true", help="print statistics")
group_c.add_argument(  # Not implemented
    "--n-negatives-per-target",
    type=int,
    default=1,
    help="number of negative samples per target",
)
group_d = parser_g.add_argument_group("dataset specific arguments")
group_d.add_argument(
    "--rating-threshold",
    type=int,
    default=4,
    help="[movielens-20m only] ratings great or equal than this are treated as valid",
)
args = parser.parse_args()

subcommand = "generate" if "logstat" in args else "download"

downloaded_datasets = _get_downloaded_datasets()
processed_datasets = _get_processed_datasets()

assert args.dataset in __datasets__, "supported datasets: {}".format(
    ", ".join(__datasets__)
)
if subcommand == "download":
    assert args.dataset not in downloaded_datasets, "you have downloaded this dataset!"
    _download(args.dataset)
else:  # generate
    assert args.dataset in downloaded_datasets, "you haven't downloaded this dataset!"
    assert (
        args.min_freq_user > args.target_len
    ), "min_freq_user should be greater than target_len"
    _generate(args)
