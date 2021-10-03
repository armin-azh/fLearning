import threading
from ._node import SingleNode


class ComputationGraphService:
    def __init__(self, parsed_yml: dict):
        self._nodes_conf = parsed_yml
        self._nodes = []
        glob_node_lock = threading.Lock()
        for key, value in self._nodes_conf.items():
            self._nodes.append(
                SingleNode(hostname=value["hostname"],
                           connections=value["connection"],
                           name=key,
                           model=None,
                           glob_lock=glob_node_lock))
