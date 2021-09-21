from .service import SyncService
from argparse import Namespace


def synchronous_service_provider(arguments: Namespace, conf: dict) -> SyncService:
    """
    create SyncService
    :param arguments:
    :param conf:
    :return:
    """
    service = SyncService.create(conf=conf, arguments=arguments)
    return service
