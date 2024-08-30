import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

class CIFAR10Dataset:
    def __init__(self, root, num_clients, train=True, transform=None):
        self.root = root
        self.num_clients = num_clients
        self.train = train
        self.transform = transform
        self.data, self.targets = self.load_data()
        self.clients = self.split_data()
        self.users = [f"user_{i}" for i in range(num_clients)]  # Add users attribute

    def load_data(self):
        dataset = datasets.CIFAR10(self.root, train=self.train, download=True, transform=self.transform)
        return dataset.data, dataset.targets

    def split_data(self):
        client_data = []
        client_size = len(self.data) // self.num_clients
        indices = np.random.permutation(len(self.data))
        for i in range(self.num_clients):
            client_indices = indices[i * client_size:(i + 1) * client_size]
            client_data.append((self.data[client_indices], np.array(self.targets)[client_indices]))
        return client_data

    def loader(self, batch_size, client_id=None, shuffle=True, seed=0):
        if client_id is None:
            dataset = torch.utils.data.TensorDataset(torch.tensor(self.data).permute(0, 3, 1, 2).float() / 255.0, torch.tensor(self.targets))
        else:
            if isinstance(client_id, str) and client_id.startswith("user_"):
                client_id = int(client_id.split("_")[1])
            client_data, client_targets = self.clients[client_id]
            client_data = torch.tensor(client_data).permute(0, 3, 1, 2).float() / 255.0  # Convert data to float type
            dataset = torch.utils.data.TensorDataset(client_data, torch.tensor(client_targets))
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def size(self, client_id=None):
        if client_id is None:
            return len(self.data)
        if isinstance(client_id, str) and client_id.startswith("user_"):
            client_id = int(client_id.split("_")[1])
        return len(self.clients[client_id][0])
    

class MNISTDataset:
    def __init__(self, root, num_clients, train=True, transform=None):
        self.root = root
        self.num_clients = num_clients
        self.train = train
        self.transform = transform
        self.data, self.targets = self.load_data()
        self.clients = self.split_data()
        self.users = [f"user_{i}" for i in range(num_clients)]  # Add users attribute

    def load_data(self):
        dataset = datasets.MNIST(self.root, train=self.train, download=True, transform=self.transform)
        return dataset.data.unsqueeze(1), dataset.targets

    def split_data(self):
        client_data = []
        client_size = len(self.data) // self.num_clients
        indices = np.random.permutation(len(self.data))
        for i in range(self.num_clients):
            client_indices = indices[i * client_size:(i + 1) * client_size]
            client_data.append((self.data[client_indices], self.targets[client_indices]))
        return client_data

    def loader(self, batch_size, client_id=None, shuffle=True, seed=0):
        if client_id is None:
            dataset = torch.utils.data.TensorDataset(self.data.float() / 255.0, self.targets)
        else:
            if isinstance(client_id, str) and client_id.startswith("user_"):
                client_id = int(client_id.split("_")[1])
            client_data, client_targets = self.clients[client_id]
            client_data = client_data.float() / 255.0  # Convert data to float type
            dataset = torch.utils.data.TensorDataset(client_data, client_targets)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def size(self, client_id=None):
        if client_id is None:
            return len(self.data)
        if isinstance(client_id, str) and client_id.startswith("user_"):
            client_id = int(client_id.split("_")[1])
        return len(self.clients[client_id][0])
