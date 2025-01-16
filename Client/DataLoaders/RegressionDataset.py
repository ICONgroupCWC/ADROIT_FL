import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
import numpy as np

class RegressionDataset(Dataset):

    def __init__(self, dataset, labels, scaler=None):

        # self.original_data = dataset
        #
        # # Initialize or use provided scaler
        # if scaler is None:
        #     self.scaler = MinMaxScaler(feature_range=(0, 1))
        #     # Reshape to 2D for fitting scaler
        #     reshaped_data = dataset.reshape(-1, dataset.shape[-1])
        #     self.scaler.fit(reshaped_data)
        # else:
        #     self.scaler = scaler
        #
        # # Transform data
        # reshaped_data = dataset.reshape(-1, dataset.shape[-1])
        # normalized_data = self.scaler.transform(reshaped_data)
        # self.dataset = normalized_data.reshape(dataset.shape)
        self.dataset = dataset
        self.labels = labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = torch.tensor(self.dataset[index]).to(torch.float32)
        label = torch.from_numpy(np.array([self.labels[index]])).to(torch.float32)
        return data, label
