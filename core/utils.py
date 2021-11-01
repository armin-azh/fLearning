import yaml
import copy
from pathlib import Path
import pickle
import numpy as np
import torch


def read_yaml_file(yaml_file: Path) -> dict:
    """
    read yaml file
    :param yaml_file:
    :return: dictionary
    """
    with open(str(yaml_file), "r") as file:
        parsed_data = yaml.load(file, Loader=yaml.FullLoader)
    return parsed_data


def select_users(n_users: int, frac: float, seed: int):
    """

    :param n_users:
    :param frac:
    :param seed:
    :return:
    """
    m = max(int(n_users * frac), 1)
    np.random.seed(seed)
    return np.random.choice(range(n_users), m, replace=False)


def fed_avg(w):
    """
    :param w: weights
    :return:
    """
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def save_parameters(args: dict, filename: Path) -> None:
    """
    save command parameter to txt file
    :param args: arguments
    :param filename: file path
    :return: None
    """
    with open(str(filename), "w") as file:
        for key, value in args.items():
            file.write(f"{key}\t{value}\n")


def send(net, conn, **kwargs):
    msg = pickle.dumps(net)
    model_ready = True
    while model_ready:
        msg = bytes(f"{len(msg):<{10}}", 'utf-8') + msg
        conn.sendall(msg)
        model_ready = False


def receive(conn, **kwargs):
    model_ready = True
    new_msg = True
    full_msg = b''
    while model_ready:
        msg = conn.recv(1024)
        if msg == b'':
            continue
        if new_msg:
            msg_len = int(msg[:10])
            new_msg = False
        full_msg += msg
        if len(full_msg) - 10 == msg_len:
            new_msg = True
            model_ready = False
            return pickle.loads(full_msg[10:])
