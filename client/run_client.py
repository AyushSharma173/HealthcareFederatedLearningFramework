# client/run_client.py
import yaml
from flwr.client import ClientApp
from client.client import MedFLClient

def load_cfg():
    return yaml.safe_load(open("client/config/client_config.yaml"))

def client_fn(_ctx):
    return MedFLClient(load_cfg()).to_client()

if __name__ == "__main__":
    cfg = load_cfg()
    # we won’t use gRPC; simulation will inject client_fn
    app = ClientApp(client_fn=client_fn)
    # app.run(...) is not used in sim
