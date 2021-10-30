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
            self._conn.append((conn, add))
            self._n_connected += 1

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



