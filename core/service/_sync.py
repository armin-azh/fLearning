from ._base import AbstractService

from core.node import Server,Client


class SyncService(AbstractService):
    def __init__(self, *args, **kwargs):
        super(SyncService, self).__init__(name="synchronous-service", type="sync")

    @classmethod
    def create(cls, **kwargs):
        conf = kwargs["conf"]
        print(conf)
