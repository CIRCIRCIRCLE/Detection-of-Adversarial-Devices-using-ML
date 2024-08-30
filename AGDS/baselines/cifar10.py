import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
import torch
import logging
from omegaconf import OmegaConf
from torchvision import transforms
import easyfl
from server import RobustServer
from baselines.client import CustomizedClient
from baselines.datasets import CIFAR10Dataset
from attacker import Lie, BitFlip, MinSum, IPM

# Customized configuration.
config_dict = {
    "attacker": {"byz_ratio": 0.2, "lie": {"lie_z": 1.5}, "bit_flip": {"flip_prob": 0.1}, "min_sum": {"min_value": -1}, "ipm": {"scale_factor": 0.01}},
    "data": {"dataset": "cifar10", "root": "./datasets", "split_type": "dir", "num_of_clients": 10},
    "server": {"rounds": 150, "clients_per_round": 10, "mode": "agas", "gas_p": 1000, "base_agg": "gas", "partitioning_method": "pcakmeans"},
    "client": {"local_epoch": 1},
    "model": "resnet18",  # Use the ResNet18 model
    "test_mode": "test_in_server",
    "gpu": 1,
}

# Convert the dictionary to an OmegaConf object
config = OmegaConf.create(config_dict)

# Register dataset.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = CIFAR10Dataset(root=config.data.root, num_clients=config.data.num_of_clients, train=True, transform=transform)
test_dataset = CIFAR10Dataset(root=config.data.root, num_clients=config.data.num_of_clients, train=False, transform=transform)

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
        device='cuda' if torch.cuda.is_available() else 'cpu'
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

# List to store test accuracies
test_accuracies = []

# Function to override the test method to record accuracies
def custom_test(self):
    test_results = self.test_in_server(self.conf.device)
    test_accuracy = test_results.get('test_accuracy', 0)
    test_accuracies.append(test_accuracy)
    print(f"Round {self._current_round} Test Accuracy: {test_accuracy:.2f}%")

# Override the test method
server.test = custom_test.__get__(server)

# Execute federated learning training
easyfl.run()