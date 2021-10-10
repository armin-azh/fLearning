from typing import Union, List, Tuple
import socket
from threading import Lock, Barrier
import threading
import time
import random
import copy
from pathlib import Path

import numpy as np

import torch
from torch.utils.data import DataLoader

from core.utils import receive, send, fed_avg


class SingleNode:
    SocketCheckList = {}
    SocketConnections = None  # nd array
    HostNameMap = {}
    HostCnt = 0
    SendIdx = None

    def __init__(self, hostname: Tuple[str, int],
                 connections: List[Tuple[str, int]],
                 name: str, model,
                 glob_lock: Lock,
                 host_idx: int,
                 save_path: Path,
                 node_type: str):
        self._node_name = name
        self._save_path = save_path.joinpath(f"node_{host_idx}")
        self._save_path.mkdir(parents=True, exist_ok=True)
        self._save_weights = self._save_path.joinpath("weight")
        self._save_weights.mkdir(parents=True, exist_ok=True)
        self._node_type = node_type

        self._node_name = f"master_{self._node_name}" if self._node_type == "master" else self._node_name

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
        self._socket_conn = []  # online connections to send
        self._master_socket_conn = []  # online master connection to send
        self._child_socket_conn = []  # online child connection to send
        self._conn = []  # online connection to receive
        self._master_conn = []  # master connection
        self._child_conn = []  # child connection
        self._socket.bind(self._hostname)
        self._socket.listen()
        self._node_type = node_type

        self._receive_models = []
        self._n_com = 0

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
            res = conn.recv(1024)
            self._conn.append((conn, add, res.decode()))
            if res.decode() == "master":
                self._master_conn.append((conn, add, res.decode()))
            else:
                self._child_conn.append((conn, add, res.decode()))
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
                sock.send(self._node_type.encode())
                add = sock.getsockname()
                add = [str(add[0]), str(add[1])]
                orig = ":".join(add)

                add = [str(conn[0]), str(conn[1])]
                des = ":".join(add)

                self._glob_node_lock.acquire()
                SingleNode.SocketCheckList[orig] = des
                self._glob_node_lock.release()

                # start, append to master or child bag
                if self._node_type == "master":
                    self._master_socket_conn.append((sock, orig))
                else:
                    self._child_socket_conn.append((sock, orig))
                # end, append to master or child bag

                self._socket_conn.append((sock, orig))
                time.sleep(random.randint(5, 9))
                _idx += 1
            except ConnectionRefusedError:
                pass

    @classmethod
    def open_access(cls):
        res = True
        for el in cls.SendIdx:
            if not isinstance(el, socket.socket):
                res = False
                break
        return res

    def send_all(self, conn, **kwargs):
        send(conn=conn, net=self._model)

    def receive_all(self, conn, _idx, add, barrier, to_master, **kwargs):
        r_model = receive(conn=conn)
        print(f"[Node({self._host_idx})] received model {_idx + 1} on {add}")
        self._lock.acquire()
        self._receive_models.append(r_model)
        self._n_com += 1
        self._lock.release()

    def exec_(self, n_round: int, epochs: int,
              train_loader: DataLoader,
              test_loader: DataLoader,
              sync_barrier: Barrier,
              agg_barrier: Barrier,
              opt, criterion,
              opt_conf):

        total_acc = []
        total_loss = []

        total_val_acc = []
        total_val_loss = []

        total_time = []

        opt = opt(self._model.parameters(), **opt_conf)

        step = 0

        _n_name = "Node" if self._node_type != "master" else "Master"
        for r in range(n_round):
            # start round
            sync_barrier.wait()
            start_time = time.time()
            print(f"[{_n_name}({self._host_idx})] starting round [{r + 1}/{n_round}]")
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

            # wait until all nodes finish their jobs
            sync_barrier.wait()
            # end of the round

            if self._node_type == "master":
                # start first, master receive child model
                print(f"[{_n_name}({self._host_idx})] ending round [{r + 1}/{n_round}] and waiting for other child")
                _idx = 0

                # start connection to receive children node
                for conn, add, _h_idx in self._child_conn:
                    t = threading.Thread(target=self.receive_all, args=(conn, _idx, add, agg_barrier, False))
                    t.start()
                    _idx += 1
                # end, connection to receive children node

                # start, make sure to receive the models
                while True:
                    self._lock.acquire()
                    if len(self._receive_models) >= len(self._child_conn):
                        self._lock.release()
                        break
                    self._lock.release()
                    time.sleep(2)
                # end, make sure to receive the models

                # start averaging with master node
                all_weights = []
                self._receive_models.append(self._model)
                for i, model in enumerate(self._receive_models):
                    w = model.state_dict()
                    all_weights.append(copy.deepcopy(w))

                avg_weights = fed_avg(all_weights)
                self._model.load_state_dict(avg_weights)
                self._receive_models = []  # reset the accumulator
                # end averaging with master node

                print(f"[{_n_name}({self._host_idx})] ending round [{r + 1}/{n_round}] received child weights.")

                # wait until all masters receive children weights and update own model
                agg_barrier.wait()

                print(f"[{_n_name}({self._host_idx})] ending round [{r + 1}/{n_round}] and waiting for other masters")
                # start connection with other masters to receive models
                _idx = 0
                for conn, add, _h_idx in self._master_conn:
                    t = threading.Thread(target=self.receive_all, args=(conn, _idx, add, agg_barrier, True))
                    t.start()
                    _idx += 1
                # end connection with other masters to receive models

                time.sleep(1)

                # start start connection to send the model
                _idx = 0
                for conn, add in self._master_socket_conn:
                    t = threading.Thread(target=self.send_all, args=(conn,))
                    t.start()
                    _idx += 1
                # end start connection to send the model

                # start make sure that masters receive the models from other masters
                while True:
                    self._lock.acquire()
                    if len(self._receive_models) >= len(self._master_conn):
                        self._lock.release()
                        break
                    self._lock.release()
                # end make sure that masters receive the models from other masters

                # start, averaging with masters models
                self._receive_models.append(self._model)
                for i, model in enumerate(self._receive_models):
                    w = model.state_dict()
                    all_weights.append(copy.deepcopy(w))

                avg_weights = fed_avg(all_weights)
                self._model.load_state_dict(avg_weights)
                # end, averaging with masters models

                # start connection to send updated models to the children
                _idx = 0
                for conn, add, _h_idx in self._child_socket_conn:
                    t = threading.Thread(target=self.send_all, args=(conn,))
                    t.start()
                    _idx += 1
                # end connection to send updated models to the children

            else:
                # start, connection to send models to master node
                time.sleep(1)
                print(f"[{_n_name}({self._host_idx})] Sending Weights to masters")
                _idx = 0
                for sock, _ in self._child_socket_conn:
                    t = threading.Thread(target=self.send_all, args=(sock,))
                    t.start()
                    _idx += 1
                # end, connection to send models to master node

                # start, connection to receive updated model from masters
                _idx = 0
                for conn, add, _h_idx in self._master_conn:
                    t = threading.Thread(target=self.receive_all, args=(conn, _idx, add, agg_barrier, True))
                    t.start()
                    _idx += 1
                # end, connection to receive updated model from masters

                # start, make sure that updated model had been received
                while True:
                    self._lock.acquire()
                    if len(self._receive_models) >= len(self._master_conn):
                        self._lock.release()
                        break
                    self._lock.release()
                print(f"[{_n_name}({self._host_idx})] Model received from master.")
                self._model = self._receive_models[0]
                self._receive_models = []
                # end, make sure that updated model had been received

            total_time.append(time.time()-start_time)
            # start testing process
            self._model.eval()
            with torch.no_grad():
                test_correct = 0
                total = 0
                for x, y in test_loader:
                    x = x.cpu()
                    y = y.cpu()
                    y_hat = self._model(x)
                    val_loss = criterion(y_hat, y)
                    prediction = torch.max(y_hat, 1)
                    total += y.size(0)
                    test_correct += np.sum(prediction[1].cpu().numpy() == y.cpu().numpy())
                val_acc = float(test_correct) / total
            total_val_acc.append(val_acc)
            total_val_loss.append(val_loss)
            # end testing process

        total_acc = np.array(total_acc)
        total_loss = np.array(total_loss)
        total_val_acc = np.array(total_val_acc)
        total_val_loss = np.array(total_val_loss)
        total_time = np.array(total_time)

        np.save(str(self._save_weights.joinpath("train_acc.npy")), total_acc)
        np.save(str(self._save_weights.joinpath("train_loss.npy")), total_loss)
        np.save(str(self._save_weights.joinpath("val_acc.npy")), total_val_acc)
        np.save(str(self._save_weights.joinpath("val_loss.npy")), total_val_loss)
        np.save(str(self._save_weights.joinpath("times.npy")), total_time)

        with open(str(self._save_weights.joinpath("n_communication.txt")), "w") as f:
            f.write(f"Number of communication: {self._n_com}")

        torch.save(self._model.state_dict(), str(self._save_weights.joinpath("model.pth")))
