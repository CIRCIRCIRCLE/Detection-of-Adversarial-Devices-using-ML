import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import easyfl
from server import RobustServer
from baselines.client import CustomizedClient
from baselines.datasets import MNISTDataset
from attacker import Lie, BitFlip, MinSum, IPM
from omegaconf import OmegaConf
from torchvision import transforms
import logging

#logging.basicConfig(level=logging.INFO)

# Customized configuration.
config_dict = {
    "attacker": {"byz_ratio": 0, "lie": {"lie_z": 1.5}, "bit_flip": {"flip_prob": 0.1}, "min_sum": {"min_value": -1}, "ipm": {"scale_factor": 0.01}},
    "data": {"dataset": "mnist", "root": "./datasets", "split_type": "dir", "num_of_clients": 10},
    "server": {"rounds": 10, "clients_per_round": 5, "mode": "agas", "gas_p": 20, "base_agg": "bulyan", "partitioning_method": "pca"},
    "client": {"local_epoch": 1},
    "model": "lenet",  # Use the out-of-the-box lenet model
    "test_mode": "test_in_server",
    "gpu": 1,
}

# Convert the dictionary to an OmegaConf object
config = OmegaConf.create(config_dict)

# Register dataset.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = MNISTDataset(root=config.data.root, num_clients=config.data.num_of_clients, train=True, transform=transform)
test_dataset = MNISTDataset(root=config.data.root, num_clients=config.data.num_of_clients, train=False, transform=transform)

easyfl.register_dataset(train_dataset, test_dataset)
easyfl.init(config)

server = RobustServer(conf=config)

# Initialize the clients
clients = []
for cid in range(config.data.num_of_clients):
    client = CustomizedClient(
        cid=f"user_{cid}",  # Ensure client id is a string
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
selected_attack = 'lie'
attack_conf = config.attacker[selected_attack]
server.set_attacker(attack_strategies[selected_attack](server.conf, server.get_byz_clients(server._clients)))

# Add debug logs
print(f"Server: {server}")
print(f"Clients: {clients}")

# Execute federated learning training
easyfl.run()
