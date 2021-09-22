from .service import SyncServerService
from argparse import Namespace


def synchronous_service_provider(arguments: Namespace, conf: dict) -> SyncServerService:
    """
    create SyncService
    :param arguments:
    :param conf:
    :return:
    """
    service = SyncServerService.create(conf=conf, arguments=arguments)
    return service
