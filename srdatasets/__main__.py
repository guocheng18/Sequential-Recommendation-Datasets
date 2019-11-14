import argparse
import logging
import sys
import unicodedata

from pandas.io.json import json_normalize
from tabulate import tabulate

from srdatasets.datasets import __datasets__
from srdatasets.download import _download
from srdatasets.process import _process
from srdatasets.utils import (
    __warehouse__,
    get_downloaded_datasets,
    get_processed_datasets,
    read_json,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d] %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)

parser = argparse.ArgumentParser("python -m srdatasets")
subparsers = parser.add_subparsers(help="commands", dest="command")
# info
parser_i = subparsers.add_parser("info", help="print local datasets info")
parser_i.add_argument("--dataset", type=str, default=None, help="dataset name")

# download
parser_d = subparsers.add_parser("download", help="download datasets")
parser_d.add_argument("--dataset", type=str, required=True, help="dataset name")

# process
parser_g = subparsers.add_parser(
    "process",
    help="process datasets",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
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
parser_g.add_argument(
    "--no-augment", action="store_true", help="Do not use data augmentation"
)
parser_g.add_argument(
    "--rating-threshold",
    type=int,
    default=4,
    help="[Amazon-X, Movielens20M, Yelp] ratings great or equal than this are treated as valid",
)
parser_g.add_argument(
    "--item-type",
    type=str,
    choices=["song", "artist"],
    default="song",
    help="[Lastfm1K] set item to song or artist",
)
args = parser.parse_args()

# Support dataset name case insensitive
_datasets_lowercase = {d.lower(): d for d in __datasets__}
if (
    "dataset" in args
    and args.dataset is not None
    and args.dataset.lower() in _datasets_lowercase
):
    args.__dict__["dataset"] = _datasets_lowercase[args.dataset.lower()]


if args.command is None:
    parser.print_help()
else:
    downloaded_datasets = get_downloaded_datasets()
    processed_datasets = get_processed_datasets()

    if args.command == "download":
        if args.dataset not in __datasets__:
            raise ValueError("Supported datasets: {}".format(", ".join(__datasets__)))
        if args.dataset in downloaded_datasets:
            raise ValueError("{} has been downloaded".format(args.dataset))
        _download(args.dataset)
    elif args.command == "process":
        if args.dataset not in downloaded_datasets:
            raise ValueError("{} has not been downloaded".format(args.dataset))
        if args.min_freq_user <= args.target_len:
            raise ValueError("min_freq_user should be greater than target_len")
        if args.dataset in processed_datasets:
            for c in processed_datasets[args.dataset]:
                config = read_json(
                    __warehouse__.joinpath(args.dataset, "processed", c, "config.json")
                )
                if all([args.__dict__[k] == v for k, v in config.items()]):
                    print("You have run this config, the config id is: {}".format(c))
                    sys.exit(1)
        _process(args)
    else:
        if args.dataset is None:
            table = [
                [
                    d,
                    unicodedata.lookup("Heavy Check Mark")
                    if d in downloaded_datasets
                    else "",
                    len(processed_datasets[d]) if d in processed_datasets else "",
                ]
                for d in __datasets__
            ]
            print(
                tabulate(
                    table,
                    headers=["name", "downloaded", "processed configs"],
                    tablefmt="psql",
                )
            )
        else:
            if args.dataset not in __datasets__:
                raise ValueError(
                    "Supported datasets: {}".format(", ".join(__datasets__))
                )
            if args.dataset not in downloaded_datasets:
                print("{} has not been downloaded".format(args.dataset))
            else:
                if args.dataset not in processed_datasets:
                    print("{} has not been processed".format(args.dataset))
                else:
                    print("Configs")
                    configs = json_normalize(
                        [
                            read_json(
                                __warehouse__.joinpath(
                                    args.dataset, "processed", c, "config.json"
                                )
                            )
                            for c in processed_datasets[args.dataset]
                        ]
                    )
                    configs.insert(0, "config id", processed_datasets[args.dataset])
                    print(
                        tabulate(
                            configs, headers="keys", showindex=False, tablefmt="psql"
                        )
                    )
                    print("\nStats")
                    stats = json_normalize(
                        [
                            read_json(
                                __warehouse__.joinpath(
                                    args.dataset, "processed", c, m, "stats.json"
                                )
                            )
                            for c in processed_datasets[args.dataset]
                            for m in ["dev", "test"]
                        ]
                    )
                    modes = ["dev", "test"] * len(processed_datasets[args.dataset])
                    stats.insert(0, "mode", modes)
                    ids = []
                    for c in processed_datasets[args.dataset]:
                        ids.extend([c, ""])
                    stats.insert(0, "config id", ids)
                    print(
                        tabulate(
                            stats, headers="keys", showindex=False, tablefmt="psql"
                        )
                    )
