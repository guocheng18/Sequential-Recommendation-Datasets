import argparse
import logging

from tabulate import tabulate

from srdatasets.datasets import __datasets__
from srdatasets.download import _download
from srdatasets.process import _process
from srdatasets.utils import _get_downloaded_datasets, _get_processed_datasets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d] %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)

parser = argparse.ArgumentParser("python -m srdatasets")
subparsers = parser.add_subparsers(help="commands", dest="command")
# info
parser_i = subparsers.add_parser("info", help="print local datasets info")

# download
parser_d = subparsers.add_parser("download", help="download datasets")
parser_d.add_argument("--dataset", type=str, required=True, help="dataset name")

# process
parser_g = subparsers.add_parser("process", help="process datasets")
parser_g.add_argument("--dataset", type=str, required=True, help="dataset name")
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
parser_g.add_argument(
    "--rating-threshold",
    type=int,
    default=4,
    help="[Amazon-X, Movielens-20M, Yelp] ratings great or equal than this are treated as valid",
)
parser_g.add_argument(
    "--item-type",
    type=str,
    default="song",
    help="[Lastfm1K] recommned artists or songs (artist | song)",
)
args = parser.parse_args()


if args.command is None:
    parser.print_help()
else:
    downloaded_datasets = _get_downloaded_datasets()
    processed_datasets = _get_processed_datasets()

    if args.command == "download":
        if args.dataset not in __datasets__:
            raise ValueError("Supported datasets: {}".format(", ".join(__datasets__)))
        if args.dataset in downloaded_datasets:
            raise ValueError("{} has been downloaded".format(args.dataset))
        _download(args.dataset)
    elif args.command == "process":
        if args.dataset not in downloaded_datasets:
            raise ValueError("{} has not been downloaded".format(args.dataset))
        if args.min_freq_user > args.target_len:
            raise ValueError("min_freq_user should be greater than target_len")
        _process(args)
    else:
        table = [
            [
                d,
                "yes" if d in downloaded_datasets else "",
                "yes" if d in processed_datasets else "",
            ]
            for d in __datasets__
        ]
        print(tabulate(table, headers=["name", "downloaded", "processed"]))
