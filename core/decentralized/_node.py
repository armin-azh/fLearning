from typing import Union, List, Tuple
import socket
from threading import Lock
import threading
import time
import random


class SingleNode:
    SocketCheckList = {}

    def __init__(self, hostname: Tuple[str, int], connections: List[Tuple[str, int]], name: str, model,
                 glob_lock: Lock):
        self._node_name = name
        self._hostname = hostname
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
