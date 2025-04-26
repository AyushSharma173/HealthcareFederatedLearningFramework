# server/server.py

from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg

def make_server_app(cfg):
    def server_fn(_ctx) -> ServerAppComponents:
        # 1) Build your aggregation strategy (swap in WFAggStrategy later)
        strategy = FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=3,
            min_available_clients=3,
        )
        # 2) Build the ServerConfig from your YAML
        config = ServerConfig(num_rounds=cfg["server"]["num_rounds"])
        # 3) Return a ServerAppComponents instance
        return ServerAppComponents(strategy=strategy, config=config)

    # 4) Wrap it in a Flower ServerApp
    return ServerApp(server_fn=server_fn)
