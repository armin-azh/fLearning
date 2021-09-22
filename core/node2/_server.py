from ._base import AbstractNode


class ServerNode(AbstractNode):
    def __init__(self, *args, **kwargs):
        super(ServerNode, self).__init__(*args, **kwargs)
