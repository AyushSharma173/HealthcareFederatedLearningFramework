# client/client.py
from flwr.client import NumPyClient
from client.local_training import load_data, train, test, get_parameters, set_parameters
from client.model_registry import load_model

class MedFLClient(NumPyClient):
    def __init__(self, cfg):
        self.net = load_model(cfg.get("model", {}))
        self.train_loader, self.val_loader = load_data(cfg.get("data", {}))
        self.epochs = cfg.get("training",{}).get("epochs",1)

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, params, config):
        set_parameters(self.net, params)
        train(self.net, self.train_loader, epochs=self.epochs)
        return get_parameters(self.net), len(self.train_loader), {}

    def evaluate(self, params, config):
        set_parameters(self.net, params)
        loss, acc = test(self.net, self.val_loader)
        return float(loss), len(self.val_loader), {"accuracy": float(acc)}
