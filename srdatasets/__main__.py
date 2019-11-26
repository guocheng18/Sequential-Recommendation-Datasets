import argparse
import logging
import os
import sys

from pandas.io.json import json_normalize
from tabulate import tabulate

from srdatasets.datasets import __datasets__
from srdatasets.download import _download
from srdatasets.process import _process
from srdatasets.utils import (__warehouse__, get_datasetname,
                              get_downloaded_datasets, get_processed_datasets,
                              read_json)

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
    "--min-freq-item", type=int, default=5, help="minimum occurrence times of item"
)
parser_g.add_argument(
    "--min-freq-user", type=int, default=5, help="minimum occurrence times of user"
)
parser_g.add_argument(
    "--task",
    type=str,
    choices=["short", "long-short"],
    default="short",
    help="short-term task or long-short-term task",
)
parser_g.add_argument(
    "--split-by",
    type=str,
    choices=["user", "time"],
    default="user",
    help="user-based or time-based dataset splitting",
)
parser_g.add_argument(
    "--dev-split",
    type=float,
    default=0.1,
    help="[user-split] the fraction of developemnt dataset",
)
parser_g.add_argument(
    "--test-split",
    type=float,
    default=0.2,
    help="[user-split] the fraction of test dataset",
)
parser_g.add_argument(
    "--input-len", type=int, default=5, help="[short] input sequence length"
)
parser_g.add_argument(
    "--target-len", type=int, default=1, help="target sequence length"
)
parser_g.add_argument(
    "--no-augment", action="store_true", help="do not use data augmentation"
)
parser_g.add_argument(
    "--remove-duplicates",
    action="store_true",
    help="remove duplicate items in user sequence",
)
parser_g.add_argument(
    "--session-interval",
    type=int,
    default=0,
    help="[short-optional, long-short-required] split user sequences into sessions (minutes)",
)
parser_g.add_argument(
    "--max-session-len", type=int, default=20, help="max session length"
)
parser_g.add_argument(
    "--min-session-len", type=int, default=2, help="min session length"
)
parser_g.add_argument(
    "--pre-sessions",
    type=int,
    default=10,
    help="[long-short] number of previous sessions",
)
parser_g.add_argument(
    "--pick-targets",
    type=str,
    choices=["last", "random"],
    default="last",
    help="[long-short] pick T random or last items from current session as targets",
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

os.makedirs(__warehouse__, exist_ok=True)

if "dataset" in args and args.dataset is not None:
    args.dataset = get_datasetname(args.dataset)


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
        if args.dataset not in __datasets__:
            raise ValueError("Supported datasets: {}".format(", ".join(__datasets__)))
        if args.dataset not in downloaded_datasets:
            raise ValueError("{} has not been downloaded".format(args.dataset))

        if args.split_by == "user":
            if args.dev_split <= 0 or args.dev_split >= 1:
                raise ValueError("dev split ratio should be in (0, 1)")
            if args.test_split <= 0 or args.test_split >= 1:
                raise ValueError("test split ratio should be in (0, 1)")

        if args.task == "short":
            if args.input_len <= 0:
                raise ValueError("input length must > 0")
            if args.session_interval < 0:
                raise ValueError("session interval must >= 0 minutes")
        else:
            if args.session_interval <= 0:
                raise ValueError("session interval must > 0 minutes")
            if args.pre_sessions < 1:
                raise ValueError("number of previous sessions must > 0")

        if args.target_len <= 0:
            raise ValueError("target length must > 0")

        if args.session_interval > 0:
            if args.min_session_len <= args.target_len:
                raise ValueError("min session length must > target length")
            if args.max_session_len < args.min_session_len:
                raise ValueError("max session length must >= min session length")

        if args.dataset in processed_datasets:
            # TODO Improve processed check when some arguments are not used
            time_splits = {}
            for c in processed_datasets[args.dataset]:
                config = read_json(
                    __warehouse__.joinpath(args.dataset, "processed", c, "config.json")
                )
                if args.split_by == "user" and all(
                    [args.__dict__[k] == v for k, v in config.items()]
                ):
                    print("You have run this config, the config id is: {}".format(c))
                    sys.exit(1)
                if args.split_by == "time" and all(
                    [
                        args.__dict__[k] == v
                        for k, v in config.items()
                        if k not in ["dev_split", "test_split"]
                    ]
                ):
                    time_splits[(config["dev_split"], config["test_split"])] = c
            args.time_splits = time_splits
        _process(args)
    else:
        if args.dataset is None:
            table = [
                [
                    d,
                    "Y" if d in downloaded_datasets else "",
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
                    print("Configs part1")
                    configs_part1 = configs.iloc[:, :8]
                    configs_part1.insert(
                        0, "config id", processed_datasets[args.dataset]
                    )
                    print(
                        tabulate(
                            configs_part1,
                            headers="keys",
                            showindex=False,
                            tablefmt="psql",
                        )
                    )
                    print("\nConfigs part2")
                    configs_part2 = configs.iloc[:, 8:]
                    configs_part2.insert(
                        0, "config id", processed_datasets[args.dataset]
                    )
                    print(
                        tabulate(
                            configs_part2,
                            headers="keys",
                            showindex=False,
                            tablefmt="psql",
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
                    modes = ["development", "test"] * len(
                        processed_datasets[args.dataset]
                    )
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
