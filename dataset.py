from torch.utils.data import Dataset
import os

class OutdoorIlluminationDataset(Dataset):
    """Dataset class -> TODO """
    def __init__(self, data_dir, transform):
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len()
    
    def __getitem__(self, index):
        return data