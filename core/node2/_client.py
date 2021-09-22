from ._base import AbstractNode


class ClientNode(AbstractNode):
    def __init__(self, *args, **kwargs):
        super(ClientNode, self).__init__(*args, **kwargs)
