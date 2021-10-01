import socket
import pickle
import numpy as np
import time
from pathlib import Path

from torch.optim import SGD

from ._base import AbstractNode
from core.trainer import ClientTrainer


class ClientNode(AbstractNode):
    def __init__(self, *args, **kwargs):
        super(ClientNode, self).__init__(*args, **kwargs)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.connect((self._ip, self._port))

    def send(self, **kwargs):
        msg = pickle.dumps(kwargs["net"])
        model_ready = True
        while model_ready:
            msg = bytes(f"{len(msg):<{10}}", 'utf-8') + msg
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

    def exec_(self, save_path: Path, **kwargs):
        n_round = kwargs["n_round"]
        rounds_acc, rounds_loss = [], []
        for c_round in range(n_round):
            print(f"[{self._id}] on round {c_round}")
            net = self.receive()
            print(f"[{self._id}] had received Net")

            opt_conf = {
                "lr": kwargs["lr"],
                "momentum": kwargs["momentum"],
                "weight_decay": kwargs["weight_decay"]
            }

            trainer = ClientTrainer(net=net, opt=SGD, opt_config=opt_conf)
            round_acc, round_loss = trainer.train(epochs=kwargs["epochs"],
                                                  train_loader=kwargs["train_loader"],
                                                  device=kwargs["device"])
            rounds_acc.append(round_acc)
            rounds_loss.append(round_loss)
            self.send(net=trainer.get_net)

        np.save(str(save_path.joinpath(f"{self._id}_acc.npy")), rounds_acc)
        np.save(str(save_path.joinpath(f"{self._id}_loss.npy")), rounds_loss)

        time.sleep(30)
        self._socket.close()
