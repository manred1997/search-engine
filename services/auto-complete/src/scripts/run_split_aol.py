import argparse
import os
import sys

from src.helpers.AOL import process
from src.utils.utils import init_logger

AUTO_COMPLETE_PATH = os.environ.get("AUTO_COMPLETE_PATH")
sys.path.append(AUTO_COMPLETE_PATH)


def main(args):
    init_logger()
    process(
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        dev_start=args.dev_start,
        dev_end=args.dev_end,
        splits=args.splits,
        columns=args.columns,
        source_dir=args.source_dir,
        target_dir=args.target_dir,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Config split
    parser.add_argument("--train_start", type=str, default="2006-03-01 00:00:00")
    parser.add_argument("--train_end", type=str, default="2006-05-18 00:00:00")
    parser.add_argument("--test_start", type=str, default="2006-05-25 00:00:00")
    parser.add_argument("--test_end", type=str, default="2006-06-01 00:00:00")
    parser.add_argument("--dev_start", type=str, default="2006-05-18 00:00:00")
    parser.add_argument("--dev_end", type=str, default="2006-05-25 00:00:00")
    parser.add_argument("--splits", type=list, default=["train", "test", "dev"])
    parser.add_argument("--columns", type=list, default=["uid", "query", "time"])

    # Path file
    parser.add_argument(
        "--source_dir",
        default=None,
        required=True,
        type=str,
        help="Path to load aol benchmark",
    )
    parser.add_argument(
        "--target_dir",
        default="../data",
        required=True,
        type=str,
        help="Path to save processed data",
    )

    args = parser.parse_args()

    main(args)
