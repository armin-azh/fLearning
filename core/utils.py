import yaml
import copy
from pathlib import Path
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
