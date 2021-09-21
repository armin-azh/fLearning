from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import argparse
from argparse import Namespace

from settings import STRATEGIES, DEFAULT_OUTPUT_DIR


def main(arguments: Namespace) -> None:
    strategy = STRATEGIES[arguments.strategy]

    if strategy == "synchronous":
        pass
    elif strategy == "asynchronous":
        pass
    elif strategy == "semi-synchronous":
        pass
    else:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help="Determine Seed", type=int, default=99)
    parser.add_argument("--out", help="output directory to collect the results", type=str, default=DEFAULT_OUTPUT_DIR)

    # system
    parser.add_argument("--strategy", help="type of strategy", type=str, choices=list(STRATEGIES.keys()),
                        default="sync")
    parser.add_argument("--n_round", help="Number of rounds", type=int, default=1)

    args = parser.parse_args()

    main(args)
