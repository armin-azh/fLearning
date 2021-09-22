class AbstractNode:
    def __init__(self, *args, **kwargs):
        self._ip = kwargs['ip']
        self._port = kwargs['port']
        self._id = kwargs["name"]

    def send(self, **kwargs):
        raise NotImplementedError

    def receive(self, **kwargs):
        raise NotImplementedError

    def exec_(self, **kwargs):
        raise NotImplementedError
