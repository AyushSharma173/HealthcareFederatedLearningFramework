# scripts/launch_simulation.py
import yaml
from flwr.simulation import run_simulation
from flwr.client import ClientApp
from server.server import make_server_app
from client.run_client import client_fn

def main():
    # 1) Load server config
    cfg = yaml.safe_load(open("server/config/server_config.yaml"))

    # 2) Build ServerApp
    server_app = make_server_app(cfg)

    # 3) Wrap your client factory in a Flower ClientApp
    client_app = ClientApp(client_fn=client_fn)

    # 4) Launch simulation with 3 “hospitals”
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=3,
        backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0.0}},
    )

if __name__ == "__main__":
    main()
