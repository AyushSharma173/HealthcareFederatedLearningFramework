import yaml
from server.server import MedFLServer

def main():
    # 1) Load server_config.yaml
    with open('server/config/server_config.yaml') as f:
        cfg = yaml.safe_load(f)

    # 2) Start API + FL server
    server = MedFLServer(cfg)
    server.start()

if __name__ == '__main__':
    main()