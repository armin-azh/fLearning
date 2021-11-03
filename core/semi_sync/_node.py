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


class ServerNode:
    SocketConnections = None  # nd array
    Accumulator = []
    OnAggregation = False
    Mapper = {}

    def __init__(self, hostname: Tuple[str, int],
                 connections: List[Tuple[str, int]],
                 model,
                 glob_lock: Lock,
                 save_path: Path):
        self._node_name = "Server"
        self._save_path = save_path.joinpath(self._node_name)
        self._save_path.mkdir(parents=True, exist_ok=True)
        self._save_weights = self._save_path.joinpath("weight")
        self._save_weights.mkdir(parents=True, exist_ok=True)

        # start make identification
        self._hostname = hostname
        # end make identification

        self._connections = connections
        self._glob_node_lock = glob_lock

        self._model_version = 0

        self._total_connections = len(self._connections)
        self._n_connected = 0

        self._model = model

        self._lock = threading.Lock()
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self._socket_conn = []  # online connections to send
        self._conn = []  # online connection to receive

        self._socket.bind(self._hostname)
        self._socket.listen()

        self._receive_models = None
        self._n_com = 0

        self._update_idx = 1

        # start connection and build graph process
        thread1 = threading.Thread(target=self._mk_accept, args=())
        thread1.start()
        time.sleep(5)
        thread2 = threading.Thread(target=self._mk_connection, args=())
        thread2.start()
        # end connection and build graph process

    def _mk_accept(self):
        _idx = 0
        while self._n_connected < self._total_connections:
            print(f"[{self._node_name}] [{self._n_connected + 1}|{self._total_connections}] Connected, "
                  f"listening on {self._hostname[0]}:{self._hostname[1]}")
            conn, add = self._socket.accept()
            self._conn.append((conn, add))
            self._n_connected += 1

            self._glob_node_lock.acquire()
            ServerNode.SocketConnections[_idx] = 1
            self._glob_node_lock.release()
            _idx += 1

        print(f"[{self._node_name}] Successfully all node are connected.")

    def _mk_connection(self):
        _idx = 0
        while _idx < self._total_connections:
            conn = self._connections[_idx]
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect(conn[0])
                add = sock.getsockname()
                add = [str(add[0]), str(add[1])]
                orig = ":".join(add)

                self._socket_conn.append((sock, orig))
                time.sleep(random.randint(5, 9))
                _idx += 1
            except ConnectionRefusedError:
                pass

    def send_all(self, conn, net, **kwargs):
        send(conn=conn, net=net)

    def receive_all(self, conn, **kwargs):
        r_model = receive(conn=conn)
        self._glob_node_lock.acquire()
        ServerNode.Accumulator.append((r_model, conn))
        self._n_com += 1
        self._glob_node_lock.release()

    def _sub_server(self, conn, n_round):

        for _idx in range(n_round):
            self.receive_all(conn)

    def exec_(self, start_barrier: Barrier, opt, criterion, opt_conf, n_round, limit, test_loader):
        print(f'[{self._node_name}] now is running')
        opt = opt(self._model.parameters(), **opt_conf)

        total_val_acc = []
        total_val_loss = []

        start_barrier.wait()

        for conn, _ in self._conn:
            t = threading.Thread(target=self._sub_server, args=(conn, n_round))
            t.start()

        while True:
            # start, check all connections are gone
            self._glob_node_lock.acquire()
            if np.all(ServerNode.SocketConnections == 0):
                self._glob_node_lock.release()
                break

            if len(ServerNode.Accumulator) >= limit:
                # start, aggregation

                # update model
                received_models = [m[0] for m in ServerNode.Accumulator]
                connections = [m[1] for m in ServerNode.Accumulator]

                all_weights = []
                received_models.append(self._model)
                for i, model in enumerate(received_models):
                    w = model.state_dict()
                    all_weights.append(copy.deepcopy(w))

                avg_weights = fed_avg(all_weights)
                self._model.load_state_dict(avg_weights)
                self._model_version += 1

                connections_to_send = []
                for conn in connections:
                    remote_port = str(conn.getpeername()[1])
                    local_port = None
                    for key, value in ServerNode.Mapper.items():
                        if value == remote_port:
                            local_port = key
                            break
                    for conn, _ in self._socket_conn:
                        r_port = str(conn.getpeername()[1])
                        if r_port == local_port:
                            connections_to_send.append(conn)
                            break

                # start, validation
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
                # end, validation

                print(
                    f"Aggregation[{self._update_idx}] | Validation Loss: {total_val_loss[-1]}, "
                    f"Validation Acc: {total_val_acc[-1]}")

                self._update_idx += 1

                # start, sending model
                for conn in connections_to_send:
                    t = threading.Thread(target=self.send_all, args=(conn, self._model))
                    t.start()
                # end, sending model
                ServerNode.Accumulator = []
                # end, aggregation

            self._glob_node_lock.release()
            # end, check all connections are gone

        total_val_acc = np.array(total_val_acc)
        total_val_loss = np.array(total_val_loss)
        np.save(str(self._save_weights.joinpath("val_acc.npy")), total_val_acc)
        np.save(str(self._save_weights.joinpath("val_loss.npy")), total_val_loss)


class ClientNode:
    def __init__(self,
                 hostname: Tuple[str, int],
                 connection: Tuple[str, int],
                 glob_lock: Lock,
                 host_idx: int,
                 save_path: Path,
                 model,
                 delay: Union[None, int]):
        self._node_name = f"Client_{host_idx}"
        self._node_idx = host_idx
        self._save_path = save_path.joinpath(self._node_name)
        self._save_path.mkdir(parents=True, exist_ok=True)
        self._save_weights = self._save_path.joinpath("weight")
        self._save_weights.mkdir(parents=True, exist_ok=True)
        self._delay = delay

        self._model = model

        self._mark = None

        self._n_com = 0

        # start make identification
        self._hostname = hostname
        # end make identification

        self._connection = connection
        self._glob_node_lock = glob_lock

        self._lock = threading.Lock()
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self._socket_conn = None  # online connections to send
        self._conn = None  # online connection to receive

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
        print(f"[{self._node_name}] Connected, listening on {self._hostname[0]}:{self._hostname[1]}")
        conn, add = self._socket.accept()
        self._conn = (conn, add)

        print(f"[{self._node_name}] Successfully all node are connected.")

    def _mk_connection(self):
        conn = self._connection
        while True:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect(conn)
                add = sock.getsockname()
                self._glob_node_lock.acquire()
                ServerNode.Mapper[str(self._hostname[1])] = str(add[1])
                self._glob_node_lock.release()
                add = [str(add[0]), str(add[1])]
                orig = ":".join(add)

                self._socket_conn = (sock, orig)
                time.sleep(random.randint(5, 9))
                break
            except ConnectionRefusedError:
                continue

    def send_all(self, conn, **kwargs):
        send(conn=conn, net=self._model)

    def receive_all(self, conn, **kwargs):
        r_model = receive(conn=conn)
        print(f"[{self._node_name}] received model.")
        self._lock.acquire()
        self._model = r_model
        self._n_com += 1
        self._lock.release()

    def exec_(self,
              n_round: int,
              epochs: int,
              train_loader: DataLoader,
              start_barrier: Barrier,
              agg_barrier: Barrier,
              opt,
              criterion,
              opt_conf):

        opt = opt(self._model.parameters(), **opt_conf)

        start_barrier.wait()

        total_acc = []
        total_loss = []
        total_time = []
        step = 0

        print(f"[{self._node_name}] is started")
        # start, training
        for r in range(n_round):
            start_time = time.time()
            print(f"[{self._node_name}] starting round [{r + 1}/{n_round}]")
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
                            f"[{self._node_name}] [{r + 1}/{n_round}] round | [{epoch + 1}/{epochs}] "
                            f"epoch | Loss: {b_loss[-1]}, Acc: {b_acc[-1]}")
                    step += 1

                ep_acc.append(np.array(b_acc).mean())
                ep_loss.append(np.array(b_loss).mean())

            total_acc.append(np.array(ep_acc).mean())
            total_loss.append(np.array(ep_loss).mean())

            # start, add delay to the training procedure
            if self._delay is not None or self._delay != 0:
                time.sleep(self._delay)
            # end, add delay to the training procedure

            # start operation
            print(f"[{self._node_name}] sending the model.")
            self.send_all(self._socket_conn[0])
            print(f"[{self._node_name}] receiving the model.")
            self.receive_all(self._conn[0])
            print(f"[{self._node_name}] model received.")
            # end operation
            total_time.append(time.time() - start_time)

        # closing connection signal
        self._glob_node_lock.acquire()
        ServerNode.SocketConnections[self._node_idx] = 0
        self._glob_node_lock.release()

        total_acc = np.array(total_acc)
        total_loss = np.array(total_loss)
        total_time = np.array(total_time)

        np.save(str(self._save_weights.joinpath("train_acc.npy")), total_acc)
        np.save(str(self._save_weights.joinpath("train_loss.npy")), total_loss)
        np.save(str(self._save_weights.joinpath("times.npy")), total_time)

        with open(str(self._save_weights.joinpath("n_communication.txt")), "w") as f:
            f.write(f"Number of communication: {self._n_com}\n")
            f.write(f"Injected Delay: {self._delay} seconds.\n")
