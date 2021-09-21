class AbstractNode:
    def __init__(self, *args, **kwargs):
        self._ip = kwargs['ip']
        self._port = kwargs['port']
        super(AbstractNode, self).__init__(*args, **kwargs)

    def send(self, **kwargs):
        raise NotImplementedError

    def receive(self, **kwargs):
        raise NotImplementedError

    def exec_(self, **kwargs):
        raise NotImplementedError

