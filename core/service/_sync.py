from typing import List
import threading

from ._base import AbstractService
from core.node import Server, Client


class SyncService(AbstractService):
    def __init__(self, serv_host: str, serv_ports: List[int], *args, **kwargs):
        self._n_clients = len(serv_ports)
        self._serv_host = serv_host
        self._serv_ports = serv_ports

        self._arguments = kwargs["arguments"]

        self._barrier = threading.Barrier(self._n_clients)
        self._server_lock = threading.Lock()

        t = threading.Thread(target=Server.aggregate, args=(self._server_lock, self._arguments))
        t.start()

        self._servers = [Server(ip=self._serv_host, port=p, arguments=self._arguments) for p in self._serv_ports]

        for serv in self._servers:
            t = threading.Thread(target=serv.exec_, args=(self._server_lock,))
            t.start()

        for _ in range(len(self._servers)):
            t.join()

        super(SyncService, self).__init__(name="synchronous-service", type="sync")

    @classmethod
    def create(cls, **kwargs):
        conf = kwargs["conf"]
        arguments = kwargs["arguments"]

        serv_conf = conf["server"]
        serv_ip = serv_conf["ip"]

        serv_ports = [int(p["port"]) for _, p in serv_conf["nodes"].items()]

        return SyncService(serv_host=serv_ip, serv_ports=serv_ports, arguments=arguments)
