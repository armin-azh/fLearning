from typing import List
import threading

from ._base import AbstractService
from core.node2 import Server, Client


class ServerSyncService(AbstractService):
    def __init__(self, serv_host: str, serv_ports: List[int], *args, **kwargs):
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
        t = threading.Thread(target=Server.aggregate, args=(self._serv_lock, self._n_round))
        t.start()

        self._servers = [Server(ip=self._serv_host, port=p, name=f"server_{idx}") for idx, p in
                         enumerate(self._serv_ports)]
        for serv in self._servers:
            t = threading.Thread(target=serv.exec_,
                                 args=(self._serv_lock, self._barrier, self._model_name, self._n_classes))
            t.start()

        for _ in range(self._total_n_clients):
            t.join()


class ClientSyncService(AbstractService):
    def __init__(self, serv_host: str, serv_port: int, client_id: str, *args, **kwargs):
        super(ClientSyncService, self).__init__(name=f"{client_id}-sync-service", type="sync")
        self._n_round = kwargs["n_round"]
        self._serv_host = serv_host
        self._serv_port = serv_port
        self._client = Client(ip=self._serv_host, port=self._serv_port, name=client_id)
        self._client.exec_(n_round=self._n_round)
