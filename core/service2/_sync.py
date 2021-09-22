from typing import List
import threading

from ._base import AbstractService
from core.node2 import Server


class ServerSyncService(AbstractService):
    local_models = []
    global_model = None

    def __init__(self, serv_host: str, serv_ports: List[int], *args, **kwargs):
        super(ServerSyncService, self).__init__(name="server-sync-service", type="sync")
        self._serv_host = serv_host
        self._serv_ports = serv_ports
