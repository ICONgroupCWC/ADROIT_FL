import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
import numpy as np

class ClassificationDataset(Dataset):

    def __init__(self, dataset, labels, scaler=None):
        self.dataset = dataset
        self.labels = labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = torch.as_tensor(self.dataset[index]).to(torch.float32)
        label = torch.tensor(self.labels[index]).type(torch.LongTensor)
        return data, label
