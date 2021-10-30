def parse_host(host: str):
    _h, _p = host.split(":")
    return _h, int(_p)


def parse_semi_sync(parsed_yml: dict) -> dict:
    nodes = {}
    _n_h, _n_p = parse_host(parsed_yml["server"]["host"])
    nodes["server"] = (_n_h, _n_p)
    nodes["fractions"] = parsed_yml["server"]["fractions"]
    nodes["limit"] = parsed_yml["server"]["limit"]
    clients = []
    for key, item in parsed_yml["server"]["clients"].items():
        _host = item["host"]
        _n_h, _n_p = parse_host(_host)
        flops = item["socket"] * item["cores"] * item["speed"] * 10e9 * item["flop"]
        clients.append(((_n_h, _n_p), flops))

    # sort the clients with by their scores
    clients.sort(key=lambda x: x[1], reverse=True)
    nodes["clients"] = clients
    return nodes
