from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import argparse
from argparse import Namespace

from settings import STRATEGIES, DEFAULT_OUTPUT_DIR, CONFIG
from core.utils import read_yaml_file
from core.provider import (sync_server_service_provider,
                           sync_client_service_provider)


def main(arguments: Namespace) -> None:
    parsed_config = read_yaml_file(CONFIG)
    strategy = parsed_config["server"]["type"]

    if strategy == "synchronous":
        if arguments.mode == "server":
            service = sync_server_service_provider(arguments=arguments, conf=parsed_config)
        elif arguments.mode == "client":
            service = sync_client_service_provider(arguments=arguments, conf=parsed_config)
        else:
            print('[Failed] Wrong mode')
    elif strategy == "asynchronous":
        pass
    elif strategy == "semi-synchronous":
        pass
    else:
        print("[Failed] Wrong ser")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="server or client node", type=str, default="server",
                        choices=["server", "client"])
    parser.add_argument("--seed", help="Determine Seed", type=int, default=99)
    parser.add_argument("--out", help="output directory to collect the results", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--client_node", help="connect the client node", type=str, default="")
    parser.add_argument("--client_loader", help="client loader", type=int, default=0)

    # system
    parser.add_argument("--n_round", help="Number of rounds", type=int, default=2)

    # local model
    parser.add_argument("--n_classes", help="number of classes", type=int, default=10, choices=[10, 100])
    parser.add_argument('--alpha', default=0.1, type=float, help="alpha for dirichlet distribution")
    parser.add_argument("--model_name", help="Model class name", type=str, default="resnet8_sm")
    parser.add_argument("--epochs", help="total number of client epochs", type=int, default=10)
    parser.add_argument('--frac', default=0.4, type=float, help="the fraction of clients: C")
    parser.add_argument("--batch_size", help="local model batch size", type=int, default=32)
    parser.add_argument("--lr", help="learning rate", type=float, default=1e-3)
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD momentum")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="SGD weight decay (default: 5e-4)")

    parser.add_argument("--n_worker", help="number of worker for load dataset", type=int, default=3)

    args = parser.parse_args()

    main(args)
