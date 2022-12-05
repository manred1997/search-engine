import argparse

from helpers.AOL import process
from utils.utils import init_logger

def main(args):
    init_logger()
    process(args)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    # Config split
    parser.add_argument('--train_start', type=str, default='2006-03-01 00:00:00')
    parser.add_argument('--train_end',   type=str, default='2006-05-18 00:00:00')
    parser.add_argument('--valid_start', type=str, default='2006-05-18 00:00:00')
    parser.add_argument('--valid_end',   type=str, default='2006-05-25 00:00:00')
    parser.add_argument('--test_start',  type=str, default='2006-05-25 00:00:00')
    parser.add_argument('--test_end',    type=str, default='2006-06-01 00:00:00')

    # Path file
    parser.add_argument("--aol_benchmark_dir", default=None, required=True, type=str, help="Path to load aol benchmark")
    parser.add_argument("--target_dir", default="../data", required=True,  type=str, help="Path to save processed data")

    args = parser.parse_args()

    main(args)