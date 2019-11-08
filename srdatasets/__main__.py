import argparse
import sys
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
subparsers = parser.add_subparsers()
# subcommmand = info
parser_i = subparsers.add_parser("info", help="print local datasets info")
parser_i.add_argument(
    "--downloaded", action="store_true", help="print downloaded datasets"
)
parser_i.add_argument(
    "--processed", action="store_true", help="print processed datasets"
)

# subcommmand = download
parser_d = subparsers.add_parser("download", help="download raw datasets")
parser_d.add_argument("--dataset", type=str, default=None, help="dataset name")

# subcommmand = generate
parser_g = subparsers.add_parser("generate", help="generate preprocessed datasets")
parser_g.add_argument("--dataset", type=str, default=None, help="dataset name")
parser_g.add_argument(
    "--dev-ratio", type=float, default=0.1, help="the fraction of developemnt dataset"
)
parser_g.add_argument(
    "--test-ratio", type=float, default=0.2, help="the fraction of test dataset"
)
parser_g.add_argument(
    "--min-freq-item", type=int, default=5, help="minimum occurrence times of item"
)
parser_g.add_argument(
    "--min-freq-user", type=int, default=10, help="minimum occurrence times of user"
)
parser_g.add_argument("--input-len", type=int, default=5, help="input sequence length")
parser_g.add_argument(
    "--target-len", type=int, default=3, help="target sequence length"
)
parser_g.add_argument("--logstat", action="store_true", help="print statistics")
parser_g.add_argument(  # Not implemented
    "--n-negatives-per-target",
    type=int,
    default=1,
    help="number of negative samples per target",
)
parser_g.add_argument(
    "--rating-threshold",
    type=int,
    default=4,
    help="[movielens-20m only] ratings great or equal than this are treated as valid",
)
parser_g.add_argument(
    "--item-type",
    type=str,
    default="song",
    help="[lastfm-1k only] recommned artists or songs (artist | song)",
)
args = parser.parse_args()


if "downloaded" in args.__dict__:
    subcommand = "info"
elif "logstat" in args.__dict__:
    subcommand = "generate"
elif len(args.__dict__) > 0:
    subcommand = "download"
else:
    parser.print_help()
    sys.exit()


downloaded_datasets = _get_downloaded_datasets()
processed_datasets = _get_processed_datasets()

if subcommand == "download":
    assert args.dataset in __datasets__, "Supported datasets: {}".format(
        ", ".join(__datasets__)
    )
    assert args.dataset not in downloaded_datasets, "You have downloaded this dataset!"
    _download(args.dataset)
elif subcommand == "generate":
    assert args.dataset in downloaded_datasets, "You haven't downloaded this dataset!"
    assert (
        args.min_freq_user > args.target_len
    ), "min_freq_user should be greater than target_len"
    _generate(args)
else:
    if not args.downloaded and not args.processed:
        parser_i.print_help()
    else:
        if args.downloaded:
            print("Downloaded datasets: {}".format(", ".join(downloaded_datasets)))
        if args.processed:
            print("Processed datasets: {}".format(", ".join(processed_datasets)))
