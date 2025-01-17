import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
import numpy as np

class RegressionDataset(Dataset):

    def __init__(self, dataset, labels, scaler=None):
        self.dataset = dataset
        self.labels = labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = torch.tensor(self.dataset[index]).to(torch.float32)
        label = torch.from_numpy(np.array([self.labels[index]])).to(torch.float32)
        return data, label
