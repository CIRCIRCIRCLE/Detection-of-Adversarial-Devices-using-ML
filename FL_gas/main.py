import easyfl
from server import RobustServer
from client import CustomizedClient
from custom_dataset import CustomFederatedDataset
from custom_model import CustomizedLSTMCNN

# Customized configuration.
config = {
    "attacker": {"byz_ratio": 0.2, "lie_z": 1.5},
    "data": {"dataset": "custom", "root": "./datasets", "split_type": "dir", "num_of_clients": 160},
    "server": {
        "rounds": 10, 
        "clients_per_round": 10, 
        "use_gas": True, 
        "gas_p": 20, 
        "base_agg": "bulyan", 
        "partitioning_method": "pca"
    },
    "client": {"local_epoch": 1},
    "model": "custom",
    "test_mode": "test_in_server",
    "gpu": 1,
}

data_paths = [
    './datasets/CIC_formatted.csv',
    './datasets/IIoT_formatted.csv',
    './datasets/IoT23_formatted.csv',
    './datasets/TON_formatted.csv'
]

num_clients_per_dataset = 40
train_dataset = CustomFederatedDataset(data_paths, num_clients=num_clients_per_dataset, train=True)
test_dataset = CustomFederatedDataset(data_paths, num_clients=num_clients_per_dataset, train=False)

# Register custom model and dataset.
easyfl.register_model(CustomizedLSTMCNN(input_dim=46, num_classes=2))
easyfl.register_dataset(train_dataset, test_dataset)

# Initialize federated learning with default configurations.
easyfl.init(config)

# Execute federated learning training. 
easyfl.run()
