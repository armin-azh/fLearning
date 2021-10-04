from typing import Union, List, Tuple
import socket
from threading import Lock, Barrier
import threading
import time
import random

import numpy as np

import torch
from torch.utils.data import DataLoader


class SingleNode:
    SocketCheckList = {}
    SocketRelease = None  # nd array
    SocketConnections = None  # nd array
    HostNameMap = {}
    HostCnt = 0

    def __init__(self, hostname: Tuple[str, int], connections: List[Tuple[str, int]], name: str, model,
                 glob_lock: Lock, host_idx: int):
        self._node_name = name

        # start make identification
        self._hostname = hostname
        self._host_idx = host_idx
        # end make identification

        self._connections = connections
        self._glob_node_lock = glob_lock
        self._total_connections = len(self._connections)
        self._n_connected = 0
        self._model = model
        self._lock = threading.Lock()
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket_conn = []
        self._conn = []
        self._socket.bind(self._hostname)
        self._socket.listen()

        # start connection and build graph process
        thread1 = threading.Thread(target=self._mk_accept, args=())
        thread1.start()
        time.sleep(5)
        thread2 = threading.Thread(target=self._mk_connection, args=())
        thread2.start()
        # end connection and build graph process

    def _mk_accept(self):
        while self._n_connected < self._total_connections:
            print(f"[{self._node_name}] [{self._n_connected + 1}|{self._total_connections}] Connected, "
                  f"listening on {self._hostname[0]}:{self._hostname[1]}")
            conn, add = self._socket.accept()
            self._conn.append(conn)
            self._n_connected += 1

        self._lock.acquire()
        SingleNode.SocketConnections[self._host_idx] = 1
        self._lock.release()
        print(f"[{self._node_name}] Successfully all node are connected.")

    def _mk_connection(self):
        _idx = 0
        while _idx < self._total_connections:
            conn = self._connections[_idx]
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect(conn)
                add = sock.getsockname()
                add = [str(add[0]), str(add[1])]
                orig = ":".join(add)

                add = [str(conn[0]), str(conn[1])]
                des = ":".join(add)

                self._glob_node_lock.acquire()
                SingleNode.SocketCheckList[orig] = des
                self._glob_node_lock.release()

                self._socket_conn.append(sock)
                time.sleep(random.randint(5, 9))
                _idx += 1
            except ConnectionRefusedError:
                pass

    # def close(self):
    #     if self._conn is not None:
    #         self._conn.close()

    def send(self, **kwargs):
        pass

    def receive(self, **kwargs):
        pass

    def exec_(self, n_round: int, epochs: int, train_loader: DataLoader, sync_barrier: Barrier, opt, criterion,
              opt_conf):
        total_acc = []
        total_loss = []

        opt = opt(self._model.parameters(), **opt_conf)

        step = 0
        for r in range(n_round):
            sync_barrier.wait()
            print(f"[Node({self._host_idx})] starting round [{r + 1}/{n_round}]")
            ep_acc = []
            ep_loss = []
            for epoch in range(epochs):
                b_acc = []
                b_loss = []
                for batch_idx, (x, y) in enumerate(train_loader):
                    x = x.cpu()
                    y = y.cpu()

                    # start updating
                    opt.zero_grad()
                    y_hat = self._model(x)
                    loss = criterion(y_hat, y)
                    loss.backward()
                    opt.step()
                    # end updating

                    pred = y_hat.data.max(1, keepdim=True)[1]
                    total_correct = pred.eq(y.data.view_as(pred)).sum()
                    curr_acc = total_correct / float(len(x))

                    b_acc.append(curr_acc.cpu())
                    b_loss.append(loss.item())

                    if step % 100 == 0 and step > 0:
                        print(
                            f"[Node({self._host_idx})] [{r + 1}/{n_round}] round | [{epoch + 1}/{epochs}] "
                            f"epoch | Loss: {b_loss[-1]}, Acc: {b_acc[-1]}")
                    step += 1

                ep_acc.append(np.array(b_acc).mean())
                ep_loss.append(np.array(b_loss).mean())

            total_acc.append(np.array(ep_acc).mean())
            total_loss.append(np.array(ep_loss).mean())

            print(f"[Node({self._host_idx})] ending round [{r + 1}/{n_round}] and waiting for other nodes")
            sync_barrier.wait()
