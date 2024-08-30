import easyfl
from server import RobustServer
from client import CustomizedClient
from custom_dataset import CustomFederatedDataset
from custom_model import CustomizedLSTMCNN
from attacker import Lie, BitFlip, MinSum, IPM
from omegaconf import OmegaConf

# Customized configuration.
config_dict = {
    "attacker": {"byz_ratio": 0, "lie": {"lie_z": 1.5}, "bit_flip": {"flip_prob": 0.1}, "min_sum": {"min_value": -1}, "ipm": {"scale_factor": 0.01}},
    "data": {"dataset": "custom", "root": "./datasets", "split_type": "dir", "num_of_clients": 100},
    "server": {"rounds": 3,"clients_per_round": 10, "mode": "agas", "gas_p": 20, "base_agg": "Bulyan", "partitioning_method": "kmeans"},
    "client": {"local_epoch": 1},
    "model": "custom",
    "test_mode": "test_in_server",
    "gpu": 1,
}

# Convert the dictionary to an OmegaConf object
config = OmegaConf.create(config_dict)

# Register custom model and dataset.
data_paths = ['./datasets/IIoT_formatted.csv']   # IIoT_formatted.csv, CIC_formatted
num_clients_per_dataset = 100
train_dataset = CustomFederatedDataset(data_paths, num_clients=num_clients_per_dataset, train=True)
test_dataset = CustomFederatedDataset(data_paths, num_clients=num_clients_per_dataset, train=False)

easyfl.register_model(CustomizedLSTMCNN(input_dim=46, num_classes=2))
easyfl.register_dataset(train_dataset, test_dataset)
easyfl.init(config)

server = RobustServer(conf=config)

# Initialize the clients
clients = []
for cid in range(config.data.num_of_clients):
    client = CustomizedClient(
        cid=cid,
        conf=config,
        train_data=train_dataset,
        test_data=test_dataset,
        device='cpu' 
    )
    clients.append(client)
server.set_clients(clients)

# Define attack strategies
attack_strategies = {
    'lie': Lie,
    'bit_flip': BitFlip,
    'min_sum': MinSum,
    'ipm': IPM
}
selected_attack = 'ipm'
attack_conf = config.attacker[selected_attack]
server.set_attacker(attack_strategies[selected_attack](server.conf, server.get_byz_clients(server._clients)))

# Execute federated learning training
easyfl.run()
