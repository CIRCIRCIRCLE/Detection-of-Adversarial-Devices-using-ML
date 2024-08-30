import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from easyfl.datasets import FederatedDataset

class CustomFederatedDataset(FederatedDataset):
    def __init__(self, data_paths, num_clients=10, train=True, test_size=0.2):
        super(CustomFederatedDataset, self).__init__()
        self.data_paths = data_paths
        self.num_clients = num_clients
        self.test_size = test_size
        self.train = train
        self.data = self._load_data()
        self._users = list(self.data.keys())

    def _load_data(self):
        data = {}
        for path in self.data_paths:
            client_base_id = path.split('/')[-1].split('_')[0]  # Base id from filename
            df = pd.read_csv(path)
            # Ensure all features are numerical
            for column in df.columns:
                if df[column].dtype == bool:
                    df[column] = df[column].astype(int)
                elif df[column].dtype == object:
                    try:
                        df[column] = pd.to_numeric(df[column])
                    except ValueError:
                        continue
            x = df.drop(columns=['label']).values
            y = df['label'].apply(lambda x: 1 if x == 'Malicious' else 0).values

            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=self.test_size, random_state=42)

            X_data = X_train if self.train else X_test
            y_data = y_train if self.train else y_test

            # Partition data for each client
            client_data = self._partition_data(X_data, y_data, client_base_id)
            data.update(client_data)

        return data

    def _partition_data(self, X_data, y_data, client_base_id):
        partitioned_data = {}
        num_samples = len(X_data)
        samples_per_client = num_samples // self.num_clients

        for i in range(self.num_clients):
            start_idx = i * samples_per_client
            end_idx = (i + 1) * samples_per_client if i < self.num_clients - 1 else num_samples

            client_id = f"{client_base_id}_client_{i}"
            partitioned_data[client_id] = {
                'x': torch.tensor(X_data[start_idx:end_idx], dtype=torch.float32),
                'y': torch.tensor(y_data[start_idx:end_idx], dtype=torch.long)
            }

        return partitioned_data

    def __len__(self):
        return sum(len(self.data[client]['x']) for client in self.data)

    def __getitem__(self, idx):
        for client in self.data:
            client_len = len(self.data[client]['x'])
            if idx < client_len:
                return self.data[client]['x'][idx], self.data[client]['y'][idx]
            idx -= client_len
        raise IndexError("Index out of range")

    def loader(self, batch_size, client_id=None, shuffle=True, seed=0):
        if client_id is None:
            data_x = torch.cat([self.data[client]['x'] for client in self.data])
            data_y = torch.cat([self.data[client]['y'] for client in self.data])
        else:
            data_x = self.data[client_id]['x']
            data_y = self.data[client_id]['y']

        dataset = TensorDataset(data_x, data_y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def size(self, client_id=None):
        if client_id:
            return len(self.data[client_id]['x'])
        else:
            return sum(len(self.data[client]['x']) for client in self.data)

    @property
    def users(self):
        return self._users
