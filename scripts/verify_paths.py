#!/usr/bin/env python3
"""Simple verifier to print resolved image paths for datasets.

Usage: run from repository root. It prints the first N resolved paths
for each dataset/list pair and whether the file exists on disk.
"""
import argparse
import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from data.data_builder import ImageDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="*", default=[], help="Pairs of root:list_path")
    parser.add_argument("--n", type=int, default=5, help="How many entries to show per dataset")
    args = parser.parse_args()

    pairs = args.pairs or [
        "./datasets/CULane:./datasets/CULane/list/train.txt",
        "./datasets/Curvelanes:./datasets/Curvelanes/train/train.txt",
    ]

    for pair in pairs:
        try:
            root, listpath = pair.split(":", 1)
        except ValueError:
            print(f"Invalid pair '{pair}', expected format root:list_path")
            continue

        print(f"\nDataset root: {root}\nList: {listpath}")
        if not os.path.exists(listpath):
            print(f"List file not found: {listpath}")
            continue

        ds = ImageDataset(root, listpath, image_size=64)
        total = len(ds)
        print(f"Total entries: {total}")
        for i in range(min(args.n, total)):
            p = ds.get_image_path(i)
            print(f" {i}: {p} -> {'EXISTS' if os.path.exists(p) else 'MISSING'}")


if __name__ == "__main__":
    main()
