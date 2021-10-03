def parse_host(host: str):
    _h, _p = host.split(":")
    return _h, int(_p)


def parse_node(parsed_yml: dict) -> dict:
    nodes = {}
    for key, item in parsed_yml.items():
        _host = item["host"]
        _n_h, _n_p = parse_host(_host)
        nodes[key] = {"hostname": (_n_h, _n_p)}
        _n_list = []
        for _host in item["connection"]:
            _n_h, _n_p = parse_host(_host)
            _n_list.append((_n_h, _n_p))
        nodes[key]["connection"] = _n_list

    return nodes
