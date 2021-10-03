from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import argparse
from argparse import Namespace

from settings import CONFIG, DEFAULT_N_ROUND, DEFAULT_N_LIMIT, RUN_TYPE
from core.utils import read_yaml_file
from core.provider import (server_service_provider,
                           client_service_provider)


def main(arguments: Namespace) -> None:
    parsed_config = read_yaml_file(CONFIG)

    if arguments.run_type == "centralized":
        strategy = parsed_config["server"]["type"]
        n_nodes = len(list(parsed_config["server"]["nodes"].keys()))
        n_limit = parsed_config["server"]["limit"]

        if n_nodes == n_limit:
            print("[Service] Synchronous")
            # synchronous
            if arguments.mode == "server":
                server_service_provider(arguments=arguments, conf=parsed_config, prefix="Synchronous")
            elif arguments.mode == "client":
                client_service_provider(arguments=arguments, conf=parsed_config, prefix="Synchronous")
            else:
                print('[Failed] Wrong mode')
        elif n_limit == 1:
            print("[Service] Asynchronous")
            # asynchronous
            if arguments.mode == "server":
                server_service_provider(arguments=arguments, conf=parsed_config, prefix="Asynchronous")
            elif arguments.mode == "client":
                client_service_provider(arguments=arguments, conf=parsed_config, prefix="Asynchronous")
            else:
                print('[Failed] Wrong mode')

        elif 1 < n_limit < n_nodes:
            print("[Service] semi- synchronous")
            # semi- synchronous
            if arguments.mode == "server":
                server_service_provider(arguments=arguments, conf=parsed_config, prefix="semi- synchronous")
            elif arguments.mode == "client":
                client_service_provider(arguments=arguments, conf=parsed_config, prefix="semi- synchronous")
            else:
                print('[Failed] Wrong mode')
        else:
            print(f"[Failed] invalid number of nodes:{n_nodes} or number of limit:{n_limit}")

    elif arguments.run_type == "decentralized":
        pass

    elif arguments.run_type == "centralized":
        pass
    else:
        print(f"[Failed] you had entered wrong option {arguments.run_type}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="server or client node", type=str, default="server",
                        choices=["server", "client"])
    parser.add_argument("--run_type", help="running type mode", type=str, default="centralized", choices=RUN_TYPE)
    parser.add_argument("--seed", help="Determine Seed", type=int, default=99)
    parser.add_argument("--client_node", help="connect the client node", type=str, default="")
    parser.add_argument("--client_loader", help="client loader", type=int, default=0)
    parser.add_argument("--run_name", help="name of the current run for split the folder", type=str,
                        default="test_3_client_semi")

    # system
    parser.add_argument("--n_round", help="Number of rounds", type=int, default=DEFAULT_N_ROUND)
    parser.add_argument("--n_limit", help="Number of round for semi-synchronous", type=int, default=DEFAULT_N_LIMIT)

    # local model
    parser.add_argument("--n_classes", help="number of classes", type=int, default=10, choices=[10, 100])
    parser.add_argument('--alpha', default=0.1, type=float, help="alpha for dirichlet distribution")
    parser.add_argument("--model_name", help="Model class name", type=str, default="cnn")
    parser.add_argument("--epochs", help="total number of client epochs", type=int, default=10)
    parser.add_argument('--frac', default=0.4, type=float, help="the fraction of clients: C")
    parser.add_argument("--batch_size", help="local model batch size", type=int, default=16)
    parser.add_argument("--lr", help="learning rate", type=float, default=1e-3)
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD momentum")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="SGD weight decay (default: 5e-4)")

    parser.add_argument("--n_worker", help="number of worker for load dataset", type=int, default=1)

    args = parser.parse_args()

    main(args)
