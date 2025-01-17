from torch.utils.data import Dataset

from DataLoaders.ImageDataset import ImageDataset
from DataLoaders.RegressionDataset import RegressionDataset
from DataLoaders.TextDataset import TextDataset


def getDataloader(dataset, labels, dataops):

    if dataops['dtype'] == 'img':
        return ImageDataset(dataset, labels, dataops)
    elif dataops['dtype'] == 'text':
        return TextDataset(dataset, labels)
    elif dataops['dtype'] == 'regression':
        return RegressionDataset(dataset, labels)

