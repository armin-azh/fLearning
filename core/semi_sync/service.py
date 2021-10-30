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

from settings import DEFAULT_OUTPUT_DIR


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
            cl_idx = np.random.choice(np.arange(len(cl)), size=n_select)

            for i in cl_idx:
                self._selected_clients.append(cl[i])
        # end, select clients

        _cu = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')

        save_path = DEFAULT_OUTPUT_DIR.joinpath("cluster").joinpath("semi-sync").joinpath(_cu)
        save_path.mkdir(exist_ok=True, parents=True)

        glob_node_lock = threading.Lock()

        server_model = create_model(name=self._model_name, num_classes=self._n_classes, device="cpu")
        server_node = ServerNode(hostname=self._nodes_conf["server"],
                                 connections=self._selected_clients,
                                 model=server_model,
                                 glob_lock=glob_node_lock,
                                 save_path=save_path)
