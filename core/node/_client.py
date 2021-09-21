import socket
from ._base import AbstractNode


class ClientNode(AbstractNode):
    def __init__(self, *args, **kwargs):
        super(ClientNode, self).__init__(*args, **kwargs)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.connect((self._ip, self._port))

    def send(self, **kwargs):
        pass

    def receive(self, **kwargs):
        pass

    def exec_(self, **kwargs):
        pass
