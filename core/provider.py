from .service import ServerService
from .service import ClientService
from argparse import Namespace

from settings import DEFAULT_OUTPUT_DIR
from core.utils import save_parameters


def server_service_provider(arguments: Namespace, conf: dict, prefix: str) -> ServerService:
    """
    create SyncService
    :param prefix:
    :param arguments:
    :param conf:
    :return:
    """
    serv_host = conf["server"]["ip"]
    serv_ports = [int(p["port"]) for p in conf["server"]["nodes"].values()]

    # _cu = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')

    save_path = DEFAULT_OUTPUT_DIR.joinpath("centralized").joinpath(prefix).joinpath(arguments.run_name).joinpath(
        "server")
    save_path.mkdir(exist_ok=True, parents=True)

    save_parameters(vars(arguments), save_path.joinpath("glob_parameters.txt"))

    service = ServerService(serv_host=serv_host,
                            serv_ports=serv_ports,
                            n_round=arguments.n_round,
                            model_name=arguments.model_name,
                            n_classes=arguments.n_classes,
                            save_path=save_path,
                            n_limit=conf["server"]["limit"])
    return service


def client_service_provider(arguments: Namespace, conf: dict, prefix:str) -> ClientService:
    """
    creat client sync service
    :param arguments:
    :param conf:
    :return:
    """
    serv_host = conf["server"]["ip"]
    serv_port = conf["server"]["nodes"][arguments.client_node]["port"]

    # _cu = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')

    save_path = DEFAULT_OUTPUT_DIR.joinpath("centralized").joinpath(prefix).joinpath(arguments.run_name).joinpath(
        "nodes")
    save_path.mkdir(exist_ok=True, parents=True)

    service = ClientService(serv_host=serv_host,
                            serv_port=serv_port,
                            client_id=arguments.client_node,
                            n_round=arguments.n_round,
                            lr=arguments.lr,
                            momentum=arguments.momentum,
                            weight_decay=arguments.weight_decay,
                            n_classes=arguments.n_classes,
                            n_clients=len(list(conf["server"]["nodes"].keys())),
                            alpha=arguments.alpha,
                            n_worker=arguments.n_worker,
                            random_seed=arguments.seed,
                            batch_size=arguments.batch_size,
                            loader_idx=arguments.client_loader,
                            epochs=arguments.epochs,
                            save_path=save_path)
    return service
