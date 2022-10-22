from torch.utils.data import Dataset
import os
import numpy as np

class OutdoorIlluminationDataset(Dataset): 

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        lst_data = os.listdir(self.data_dir)

        lst_input = [f for f in lst_data if f.startswith('input')] 
        lst_albedo = [f for f in lst_data if f.startswith('albedo')] 
        lst_shading = [f for f in lst_data if f.startswith('shading')]
        
        lst_input.sort()
        lst_albedo.sort()
        lst_shading.sort()

        self.lst_input = lst_input
        self.lst_albedo = lst_albedo
        self.lst_shading = lst_shading

    def __len__(self):
        assert len(self.lst_input) == len(self.lst_albedo)
        assert len(self.lst_input) == len(self.lst_shading)

        return len(self.lst_input)
	

    def __getitem__(self, index):
        inputs = np.load(os.path.join(self.data_dir, self.lst_input[index]))
        albedos = np.load(os.path.join(self.data_dir, self.lst_albedo[index]))
        shadings = np.load(os.path.join(self.data_dir, self.lst_shading[index]))

        inputs = (inputs/255.0).astype(np.float32)
        albedos = (albedos/255.0).astype(np.float32)
        shadings = (shadings/255.0).astype(np.float32)
    
        data = {'input':inputs, 'albedo':albedos, 'shading':shadings}

        if self.transform:				
            data = self.transform(data)

        return data