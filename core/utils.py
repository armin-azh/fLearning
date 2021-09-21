import yaml
from pathlib import Path


def read_yaml_file(yaml_file: Path) -> dict:
    """
    read yaml file
    :param yaml_file:
    :return: dictionary
    """
    with open(str(yaml_file), "r") as file:
        parsed_data = yaml.load(file, Loader=yaml.FullLoader)
    return parsed_data
