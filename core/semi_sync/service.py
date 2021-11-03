import warnings

warnings.filterwarnings("ignore")

import threading
import numpy as np
import time
from datetime import datetime
import torch
from torch import nn

from core.models.model_factory import create_model
from core.loader import get_cifar, get_loaders
from ._node import ServerNode
from ._node import ClientNode

from settings import DEFAULT_OUTPUT_DIR, TIME_SCALE
from core.utils import save_parameters


class SemiSyncComputationGraphService:
    def __init__(self, parsed_yml: dict, model_name: str, n_classes: int):
        self._nodes_conf = parsed_yml
        self._model_name = model_name
        self._n_classes = n_classes
        self._clients = parsed_yml["clients"]
        self._fractions = parsed_yml["fractions"]
        self._clients_classes = []
        self._clients_batch = int(len(self._clients) / len(self._fractions))
        idx = 0
        while idx < len(self._clients):
            self._clients_classes.append(self._clients[idx:idx + self._clients_batch])
            idx += self._clients_batch

        # start, select clients
        self._selected_clients = []
        for idx in range(len(self._fractions)):
            frac = self._fractions[idx]
            cl = self._clients_classes[idx]
            n_select = int(np.ceil(frac * len(cl)))
            cl_idx = np.random.choice(np.arange(len(cl)), size=n_select, replace=False)

            for i in cl_idx:
                self._selected_clients.append(cl[i])
        # end, select clients

        _cu = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')

        save_path = DEFAULT_OUTPUT_DIR.joinpath("cluster").joinpath("semi-sync").joinpath(_cu)
        save_path.mkdir(exist_ok=True, parents=True)

        self._save_path = save_path

        ServerNode.SocketConnections = np.zeros((len(self._selected_clients),))  # initiate the connection
        glob_node_lock = threading.Lock()

        server_model = create_model(name=self._model_name, num_classes=self._n_classes, device="cpu")
        self._server_node = ServerNode(hostname=self._nodes_conf["server"],
                                       connections=self._selected_clients,
                                       model=server_model,
                                       glob_lock=glob_node_lock,
                                       save_path=save_path)

        self._nodes = []
        idx = 0

        flops = np.array([p for (_, _), p in self._selected_clients])

        for val in self._selected_clients:
            (host, port), flop = val
            delay_fraction = np.abs(flop - flops.mean()) / (flops.std() + np.finfo(np.float32).eps)

            client_model = create_model(name=self._model_name, num_classes=self._n_classes, device="cpu")
            self._nodes.append(ClientNode(hostname=(host, port),
                                          connection=self._nodes_conf["server"],
                                          glob_lock=glob_node_lock,
                                          host_idx=idx,
                                          save_path=save_path,
                                          model=client_model,
                                          delay=np.ceil(delay_fraction * TIME_SCALE).astype(np.int)))
            idx += 1

        # start, make sure that all computation graph is built
        while True:
            glob_node_lock.acquire()
            if np.all(ServerNode.SocketConnections == 1):
                glob_node_lock.release()
                break
            glob_node_lock.release()
            time.sleep(3)  # bigger number for bigger computation graph
        # end, make sure that all computation graph is built

    def train(self, arguments):
        # start training process

        save_parameters(args=vars(arguments), filename=self._save_path.joinpath("parameters.txt"))
        print(f"[Train] now start training process on {self._n_classes} nodes")

        start_barrier = threading.Barrier(parties=len(self._nodes) + 1)
        agg_barrier = threading.Barrier(parties=self._nodes_conf["limit"])

        opt_conf = {
            "lr": arguments.lr,
            "momentum": arguments.momentum,
            "weight_decay": arguments.weight_decay
        }

        train_dataset, test_dataset = get_cifar(self._n_classes)
        client_loaders, test_loader = get_loaders(train_dataset,
                                                  test_dataset,
                                                  n_clients=len(self._nodes),
                                                  alpha=arguments.alpha,
                                                  batch_size=arguments.batch_size,
                                                  n_data=None,
                                                  num_workers=arguments.n_worker,
                                                  seed=arguments.seed)

        # start, start all client node
        for idx, node in enumerate(self._nodes):
            opt = torch.optim.SGD
            loss = nn.CrossEntropyLoss()
            train_loader = client_loaders[idx]
            t = threading.Thread(target=node.exec_, args=(arguments.n_round,
                                                          arguments.epochs,
                                                          train_loader,
                                                          start_barrier,
                                                          agg_barrier,
                                                          opt,
                                                          loss,
                                                          opt_conf))
            t.start()
        # end, start all client node

        # start, start server node
        opt = torch.optim.SGD
        loss = nn.CrossEntropyLoss()
        t = threading.Thread(target=self._server_node.exec_,
                             args=(start_barrier,
                                   opt, loss,
                                   opt_conf,
                                   arguments.n_round,
                                   self._nodes_conf["limit"],
                                   test_loader))
        t.start()
        # end, start server node
