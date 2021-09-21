from threading import Lock
from argparse import Namespace
from ._base import AbstractNode


class ServerNode(AbstractNode):
    def __init__(self, *args, **kwargs):
        super(ServerNode, self).__init__(*args, **kwargs)

    def send(self, **kwargs):
        pass

    def receive(self, **kwargs):
        pass

    def exec_(self, **kwargs):
        pass

    def aggregate(self, lock: Lock, arguments: Namespace):
        pass
