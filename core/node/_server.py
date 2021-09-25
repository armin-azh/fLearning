import copy
import time
import socket
import pickle
from pathlib import Path
from threading import Lock, Barrier

import numpy as np

from ._base import AbstractNode
from core.utils import fed_avg
from core.trainer import eval_global_model
from core.normalizer import CIFAR_TRANSFORMER

import torch
import torchvision
from torch.utils.data import DataLoader


class ServerNode(AbstractNode):
    local_models = []
    global_model = None
    n_local_models = 0
    total_n_worker = 0
    release = 0
    n_connected = 0

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

    def exec_(self, lock: Lock, barrier: Barrier, n_round: int, save_path: Path, **kwargs):
        self.connect()
        ServerNode.n_connected += 1

        # send model to the client
        self.send(net=ServerNode.global_model)

        time_cont = []

        for c_round in range(n_round):

            s_time = time.time()

            # ServerNode.local_models = []

            client_model = self.receive()  # receive model from client

            # update global variable
            lock.acquire()
            ServerNode.local_models.append(client_model)
            ServerNode.n_local_models += 1

            # modify last round
            if c_round + 1 == n_round:
                ServerNode.total_n_worker -= 1
                ServerNode.n_connected -= 1

            lock.release()

            # waiting for aggregating
            while True:
                print(f"[{self._id}] is waiting for updated model")

                lock.acquire()
                if ServerNode.release == 1:
                    lock.release()
                    break
                lock.release()
                time.sleep(1)

            e_time = time.time() - s_time
            time_cont.append(int(e_time))
            print(f"[{self._id}] Took {e_time} sec")
            np.save(str(save_path.joinpath(f"{self._id}_round_times.npy")), np.array(time_cont))
            self.send(net=ServerNode.global_model)

            barrier.wait()
            ServerNode.release = 0

        # close socket
        self._socket.close()

    @classmethod
    def aggregate(cls, lock: Lock, n_round: int, save_path: Path, n_limit: int, *args, **kwargs):
        model_path = save_path.joinpath("model")
        model_path.mkdir(exist_ok=True, parents=True)

        val_glob_acc_cont = []
        val_glob_loss_cont = []

        # update
        update_idx = 0
        while cls.total_n_worker > 0:

            # check all worker are arrived
            while True:
                print(f"[Accumulator] Waiting for clients | locals: {cls.n_local_models}, total: {cls.total_n_worker}, "
                      f"connected: {cls.n_connected}")
                lock.acquire()
                if n_limit <= cls.n_local_models:
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

            torch.save(cls.global_model.state_dict(), str(model_path.joinpath(f"global_model_r_{update_idx + 1}.pth")))

            # global validation
            test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                                    transform=CIFAR_TRANSFORMER)
            test_loader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=2)

            val_glob_acc, val_glob_loss = eval_global_model(net=cls.global_model, test_loader=test_loader)
            val_glob_acc_cont.append(val_glob_acc)
            val_glob_loss_cont.append(val_glob_loss)

            print(f"[Accumulator] Update [{update_idx + 1}] | Val Acc: {val_glob_acc}, Val Loss: {val_glob_loss}")

            update_idx += 1

            lock.release()
            cls.release = 1

        # save values
        np.save(str(save_path.joinpath("val_glob_acc.npy")), np.array(val_glob_acc_cont))
        np.save(str(save_path.joinpath("val_glob_loss.npy")), np.array(val_glob_loss_cont))
