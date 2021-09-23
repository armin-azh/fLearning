from typing import List
import threading
from pathlib import Path

from ._base import AbstractService
from core.node import Server, Client
from core.models.model_factory import create_model
from core.loader import get_cifar, get_loaders


class ServerSyncService(AbstractService):
    def __init__(self, serv_host: str, serv_ports: List[int], save_path: Path, *args, **kwargs):
        super(ServerSyncService, self).__init__(name="server-sync-service", type="sync")
        self._n_round = kwargs["n_round"]
        self._model_name = kwargs["model_name"]
        self._n_classes = kwargs["n_classes"]

        self._serv_host = serv_host
        self._serv_ports = serv_ports
        self._total_n_clients = len(self._serv_ports)

        self._barrier = threading.Barrier(self._total_n_clients)
        self._serv_lock = threading.Lock()
        Server.total_n_worker = self._total_n_clients
        Server.global_model = create_model(name=self._model_name, num_classes=self._n_classes, device=0)
        t = threading.Thread(target=Server.aggregate, args=(self._serv_lock, self._n_round, save_path))
        t.start()

        self._servers = [Server(ip=self._serv_host, port=p, name=f"server_{idx}") for idx, p in
                         enumerate(self._serv_ports)]
        for serv in self._servers:
            t = threading.Thread(target=serv.exec_,
                                 args=(
                                     self._serv_lock, self._barrier, self._n_round, save_path))
            t.start()

        for _ in range(self._total_n_clients):
            t.join()


class ClientSyncService(AbstractService):
    def __init__(self, serv_host: str, serv_port: int, client_id: str, save_path: Path, *args, **kwargs):
        super(ClientSyncService, self).__init__(name=f"{client_id}-sync-service", type="sync")
        self._n_round = kwargs["n_round"]
        self._n_classes = kwargs["n_classes"]
        self._n_clients = kwargs["n_clients"]
        self._serv_host = serv_host
        self._serv_port = serv_port

        train_dataset, test_dataset = get_cifar(self._n_classes)
        client_loaders, test_loader = get_loaders(train_dataset,
                                                  test_dataset,
                                                  n_clients=self._n_clients,
                                                  alpha=kwargs["alpha"],
                                                  batch_size=kwargs["batch_size"],
                                                  n_data=None,
                                                  num_workers=kwargs["n_worker"],
                                                  seed=kwargs["random_seed"], )

        self._client = Client(ip=self._serv_host, port=self._serv_port, name=client_id)
        self._client.exec_(n_round=self._n_round,
                           lr=kwargs["lr"],
                           momentum=kwargs["momentum"],
                           weight_decay=kwargs["weight_decay"],
                           device=0,
                           train_loader=client_loaders[kwargs["loader_idx"]],
                           epochs=kwargs["epochs"],
                           )
