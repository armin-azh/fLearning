from typing import List
from ._base import AbstractService

from core.node import Server, Client


class SyncService(AbstractService):
    def __init__(self, serv: Server, nodes: List[Client], *args, **kwargs):
        self._n_clients = len(nodes)
        self._server = serv
        self._clients = nodes
        super(SyncService, self).__init__(name="synchronous-service", type="sync")

    @classmethod
    def create(cls, **kwargs):
        conf = kwargs["conf"]
        serv_conf = conf["server"]
        serv_name = serv_conf["name"]
        serv_ip = serv_conf["ip"]
        serv_port = int(serv_conf["port"])
        clients = []

        serv = Server(ip=serv_ip,port=serv_port)

        for key, value in serv_conf["nodes"].items():
            _ip = value["ip"]
            _port = int(value["port"])

            _n = Client(ip=_ip, port=_port)
            clients.append(_n)

        return cls(serv=serv, nodes=clients)
