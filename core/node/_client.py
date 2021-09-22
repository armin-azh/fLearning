import socket
import pickle
from ._base import AbstractNode


class ClientNode(AbstractNode):
    def __init__(self, *args, **kwargs):
        super(ClientNode, self).__init__(*args, **kwargs)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.connect((self._ip, self._port))
        self._arguments = kwargs["arguments"]

    def send(self, **kwargs):
        msg = pickle.dumps(kwargs["net"])
        model_ready = True
        while model_ready:
            msg = bytes(f"{len(msg):<{10}}", 'utf-8')+msg
            self._socket.sendall(msg)
            model_ready = False

    def receive(self, **kwargs):
        model_ready = True
        new_msg = True
        full_msg = b''
        while model_ready:
            msg = self._socket.recv(1024)
            if new_msg:
                msg_len = int(msg[:10])
                new_msg = False
            full_msg += msg
            if len(full_msg) - 10 == msg_len:
                new_msg = True
                model_ready = False
                return pickle.loads(full_msg[10:])

    def exec_(self, **kwargs):

        for c_round in range(self._arguments.n_round):
            print(f"[Client] on round {c_round}")

            net = self.receive()
            print(f"[Client] had received Net")
