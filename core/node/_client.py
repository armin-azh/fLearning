from ._base import AbstractNode


class ClientNode(AbstractNode):
    def __init__(self, *args, **kwargs):
        super(ClientNode, self).__init__(*args, **kwargs)

    def send(self, **kwargs):
        pass

    def receive(self, **kwargs):
        pass

    def exec_(self, **kwargs):
        pass
