from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image

class OutdoorIlluminationDataset(Dataset): 

    # TODO: metadata loading part
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        lst_input = os.listdir(os.path.join(self.data_dir, 'original'))
        lst_input.sort()
        lst_mask = os.listdir(os.path.join(self.data_dir, 'environment'))
        lst_mask.sort()
        lst_albedo = os.listdir(os.path.join(self.data_dir, 'albedo'))
        lst_albedo.sort()
        lst_shading = os.listdir(os.path.join(self.data_dir, 'shading'))
        lst_shading.sort()

        lst_input.sort()
        lst_mask.sort()
        lst_albedo.sort()
        lst_shading.sort()

        self.lst_input = lst_input
        self.lst_mask = lst_mask
        self.lst_albedo = lst_albedo
        self.lst_shading = lst_shading

    def __len__(self):
        assert len(self.lst_input) == len(self.lst_mask)
        assert len(self.lst_input) == len(self.lst_albedo)
        assert len(self.lst_input) == len(self.lst_shading)
        
        return len(self.lst_input)
	

    def __getitem__(self, index):
        inputs = Image.open(os.path.join(self.data_dir, 'original', self.lst_input[index]))
        masks = Image.open(os.path.join(self.data_dir, 'environment', self.lst_mask[index]))
        albedos = Image.open(os.path.join(self.data_dir, 'albedo', self.lst_albedo[index]))
        shadings = Image.open(os.path.join(self.data_dir, 'shading', self.lst_shading[index]))

        if self.transform:				
            inputs = self.transform(inputs)
            masks = self.transform(masks)
            albedos = self.transform(albedos)
            shadings = self.transform(shadings)

        data = {'input':inputs, 
        'mask':masks, 
        'albedo':albedos, 
        'shading':shadings}

        return data