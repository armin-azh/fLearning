import threading
from ._node import SingleNode
from core.models.model_factory import create_model
import numpy as np
import time


class ComputationGraphService:
    def __init__(self, parsed_yml: dict, model_name: str, n_classes: int):
        self._nodes_conf = parsed_yml
        self._model_name = model_name
        self._n_classes = n_classes
        self._nodes = []

        n_nodes = len(list(self._nodes_conf.values()))  # number of nodes
        SingleNode.SocketRelease = np.zeros((n_nodes,))  # initiate the release tab
        SingleNode.SocketConnections = np.zeros((n_nodes,))  # initiate the connection
        glob_node_lock = threading.Lock()

        # start initiate node and build the computation graph
        for key, value in self._nodes_conf.items():
            n_model = create_model(name=self._model_name, num_classes=self._n_classes, device="cpu")

            joined_hostname = ":".join([value["hostname"][0], str(value["hostname"][1])])
            SingleNode.HostNameMap[joined_hostname] = SingleNode.HostCnt

            self._nodes.append(
                SingleNode(hostname=value["hostname"],
                           connections=value["connection"],
                           name=key,
                           model=n_model,
                           glob_lock=glob_node_lock,
                           host_idx=SingleNode.HostCnt))
            SingleNode.HostCnt += 1
        # end initiate node and build the computation graph

        # start, make sure that all computation graph is built
        while True:
            glob_node_lock.acquire()
            if np.all(SingleNode.SocketConnections == 1):
                glob_node_lock.release()
                break
            glob_node_lock.release()
            time.sleep(3)  # bigger number for bigger computation graph
        # end, make sure that all computation graph is built

        # start training process
        print(f"[Train] now start training process")

        print(SingleNode.SocketConnections)
        # end training process
