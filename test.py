import argparse

parser = argparse.ArgumentParser("python -m srdatasets")

parser.add_argument("--dataset", type=str, default="test")

args = parser.parse_args()

print(args)

print("dataset" in args)

args.__dict__["dataset"] = "changed"

print(args)