from .service import SyncServerService, SyncClientService
from argparse import Namespace


def sync_server_service_provider(arguments: Namespace, conf: dict) -> SyncServerService:
    """
    create SyncService
    :param arguments:
    :param conf:
    :return:
    """
    service = SyncServerService.create(conf=conf, arguments=arguments)
    return service


def sync_client_service_provider(arguments: Namespace, conf: dict) -> SyncClientService:
    """
    creat client sync service
    :param arguments:
    :param conf:
    :return:
    """
    service = SyncClientService.create(conf=conf, arguments=arguments)
    return service
