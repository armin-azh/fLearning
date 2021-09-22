import copy
import time
import socket
import pickle
from pathlib import Path
from threading import Lock, Barrier
from argparse import Namespace
from ._base import AbstractNode
from core.utils import fed_avg
import torch


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

    def send(self, **kwargs):
        msg = pickle.dumps(kwargs["net"])
        model_ready = True
        while model_ready:
            msg = bytes(f"{len(msg):<{10}}", 'utf-8') + msg
            self._conn.sendall(msg)
            model_ready = False

    def receive(self, **kwargs):
        model_ready = True
        new_msg = True
        full_msg = b''
        while model_ready:
            msg = self._conn.recv(1024)
            if new_msg:
                msg_len = int(msg[:10])
                new_msg = False
            full_msg += msg
            if len(full_msg) - 10 == msg_len:
                new_msg = True
                model_ready = False
                return pickle.loads(full_msg[10:])

    def exec_(self, lock: Lock, barrier: Barrier, n_round: int, **kwargs):
        self.connect()

        # send model to the client
        self.send(net=ServerNode.global_model)

        for c_round in range(n_round):

            ServerNode.local_models = []

            client_model = self.receive()  # receive model from client

            # update global variable
            lock.acquire()
            ServerNode.local_models.append(client_model)
            ServerNode.n_local_models += 1
            lock.release()

            # waiting for aggregating
            while True:
                print(f"[{self._id}] is waiting for aggregating model")

                lock.acquire()
                if ServerNode.release == 1:
                    lock.release()
                    break
                lock.release()
                time.sleep(1)

            self.send(net=ServerNode.global_model)
            barrier.wait()
            ServerNode.release = 0

        # close socket
        self._socket.close()

    @classmethod
    def aggregate(cls, lock: Lock, n_round: int, save_path: Path, *args, **kwargs):
        model_path = save_path.joinpath("model")
        model_path.mkdir(exist_ok=True,parents=True)

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
            all_weights = []

            for i, model in enumerate(cls.local_models):
                w = model.state_dict()
                all_weights.append(copy.deepcopy(w))
            print("[Accumulator] Collected weights")

            avg_weights = fed_avg(all_weights)
            cls.global_model.load_state_dict(avg_weights)
            # release the sources

            torch.save(cls.global_model.state_dict(), str(model_path.joinpath(f"round_{c_round+1}.pth")))

            lock.release()
            cls.release = 1


