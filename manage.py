from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import argparse
from argparse import Namespace

from settings import STRATEGIES, DEFAULT_OUTPUT_DIR, CONFIG
from core.utils import read_yaml_file
from core.provider import synchronous_service_provider


def main(arguments: Namespace) -> None:
    parsed_config = read_yaml_file(CONFIG)
    strategy = parsed_config["server"]["type"]

    if strategy == "synchronous":
        service = synchronous_service_provider(arguments=arguments, conf=parsed_config)
    elif strategy == "asynchronous":
        pass
    elif strategy == "semi-synchronous":
        pass
    else:
        print("[Failed] Wrong ser")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help="Determine Seed", type=int, default=99)
    parser.add_argument("--out", help="output directory to collect the results", type=str, default=DEFAULT_OUTPUT_DIR)

    # system
    parser.add_argument("--n_round", help="Number of rounds", type=int, default=1)

    args = parser.parse_args()

    main(args)
