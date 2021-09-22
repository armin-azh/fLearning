import threading
from .service2 import ServerSyncService
from .service2 import ClientSyncService
from argparse import Namespace


def sync_server_service_provider(arguments: Namespace, conf: dict) -> ServerSyncService:
    """
    create SyncService
    :param arguments:
    :param conf:
    :return:
    """
    serv_host = conf["server"]["ip"]
    serv_ports = [int(p["port"]) for p in conf["server"]["nodes"].values()]

    service = ServerSyncService(serv_host=serv_host, serv_ports=serv_ports, n_round=arguments.n_round)
    return service


def sync_client_service_provider(arguments: Namespace, conf: dict) -> ClientSyncService:
    """
    creat client sync service
    :param arguments:
    :param conf:
    :return:
    """
    serv_host = conf["server"]["ip"]
    serv_port = conf["server"]["nodes"][arguments.client_node]["port"]
    service = ClientSyncService(serv_host=serv_host,
                                serv_port=serv_port,
                                client_id=arguments.client_node,
                                n_round=arguments.n_round)
    return service
