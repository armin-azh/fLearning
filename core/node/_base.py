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


class ServerNode(AbstractNode):
    def __init__(self, *args, **kwargs):
        super(ServerNode, self).__init__(*args, **kwargs)

    def send(self):
        pass

    def receive(self):
        pass

    def exec_(self, **kwargs):
        pass


class ClientNode(AbstractNode):
    def __init__(self, *args, **kwargs):
        super(ClientNode, self).__init__(*args, **kwargs)

    def send(self):
        pass

    def receive(self):
        pass

    def exec_(self, **kwargs):
        pass
