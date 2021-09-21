import time
import copy
from threading import Lock
from argparse import Namespace
import pickle
import socket

import torch
import torchvision
from torch.utils.data import DataLoader

from ._base import AbstractNode
from core.utils import select_users, fed_avg
from core.models.model_factory import create_model
from core.normalizer import CIFAR_TRANSFORMER


class ServerNode(AbstractNode):
    num_local_models = 0
    local_models = []
    total_workers = 10
    global_model = 0
    release = 0

    def __init__(self, *args, **kwargs):
        super(ServerNode, self).__init__(*args, **kwargs)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((self._ip, self._port))
        self._socket.listen()

        self._conn, _ = self._socket.accept()
        self._arguments = kwargs["arguments"]

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

    def exec_(self, lock: Lock, **kwargs):

        ServerNode.global_model = create_model(name=self._arguments.model_name, device=0)

        self.send(net=ServerNode.global_model)

        for c_round in range(self._arguments.n_round):
            t1 = time.time()
            ServerNode.local_models = []

            worker_model = self.receive()

            lock.acquire()
            ServerNode.local_models.append(worker_model)
            ServerNode.num_local_models += 1
            lock.release()

            while True:
                print(f"[Waiting] Server Thread is Waiting")

                lock.acquire()
                if ServerNode.release == 1:
                    lock.release()
                    break
                lock.release()
                time.sleep(1)

            self.send(net=ServerNode.global_model)

        self._socket.close()

    @classmethod
    def aggregate(cls, lock: Lock, arguments: Namespace):

        for c_round in range(arguments.n_round):

            while True:
                print(f"[Waiting] For a worker")
                lock.acquire()

                if cls.total_workers == cls.num_local_models:
                    cls.num_local_models = 0
                    break

                lock.release()
                time.sleep(1)

            part_users = select_users(n_users=cls.total_workers, frac=arguments.frac, seed=arguments.seed)

            all_weights = []
            edge_models = []

            for i, model in enumerate(cls.local_models):
                if i in part_users:
                    w = model.state_dict()
                    edge_models.append(model)
                    all_weights.append(copy.deepcopy(w))

            averaged_weights = fed_avg(all_weights)
            cls.global_model.load_state_dict(averaged_weights)

            test_set = torchvision.datasets.CIFAR10(root='./data',
                                                    train=False,
                                                    download=True,
                                                    transform=CIFAR_TRANSFORMER)

            test_loader = DataLoader(test_set, batch_size=100, shuffle=False, num_workers=arguments.n_worker)

            cls.release = 1
            lock.release()
