from DataLoaders.TextDataset import TextDataset
from DataLoaders.ImageDataset import ImageDataset
from DataLoaders.RegressionDataset import RegressionDataset
from DataLoaders.ClassificationDataset import ClassificationDataset

def getDataloader(dataset, labels, dataops):

    '''Selecting data loader according to the data type'''

    if dataops['dtype'] == 'img':
        return ImageDataset(dataset, labels, dataops)
    elif dataops['dtype'] == 'text':
        return TextDataset(dataset, labels)
    elif dataops['dtype'] == 'regression':
        return RegressionDataset(dataset, labels)
    elif dataops['dtype'] == 'classification':
        return ClassificationDataset(dataset, labels)
