import time
import socket
from threading import Lock, Barrier
from argparse import Namespace
from ._base import AbstractNode
from core.models.model_factory import create_model


class ServerNode(AbstractNode):
    local_models = []
    global_model = None
    n_local_models = 0
    total_n_worker = 0
    release = 0

    def __init__(self, *args, **kwargs):
        super(ServerNode, self).__init__(*args, **kwargs)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._conn = None
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def connect(self):
        print(f"[{self._id}] is on the thread ...")
        print(f"[{self._id}] is listening on the {self._ip}:{self._port}")
        self._socket.bind((self._ip, self._port))
        self._socket.listen()
        self._conn, _ = self._socket.accept()
        print(f"[{self._id}] client connected on the {self._ip}:{self._port}")

    def send(self):
        pass

    def receive(self, **kwargs):
        pass

    def exec_(self, lock: Lock, barrier: Barrier, model_name: str, n_classes: int, **kwargs):
        self.connect()

        if ServerNode.global_model is None:
            ServerNode.global_model = create_model(name=model_name, num_classes=n_classes, device=0)

        barrier.wait()

    @classmethod
    def aggregate(cls, lock: Lock, n_round: int, *args, **kwargs):

        for c_round in range(n_round):

            # check all worker are arrived
            while True:
                print(f"[Accumulator] Waiting for clients")
                print(f"locals: {cls.n_local_models}, total: {cls.total_n_worker}")
                lock.acquire()
                if cls.total_n_worker == cls.n_local_models:
                    cls.n_local_models = 0
                    break
                lock.release()
                time.sleep(1)

            # Do something

            # release the sources

            lock.release()
            cls.release = 1
