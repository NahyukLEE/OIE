from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image

class OutdoorIlluminationDataset(Dataset): 

    # TODO: metadata loading part
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        lst_data = os.listdir(self.data_dir)
        lst_data.sort()
        lst_input = [f for f in lst_data][:5526] #[f for f in lst_data if 'input' in f] 
        #lst_mask = [f for f in lst_data if 'mask' in f] 
        lst_albedo = [f for f in lst_data][5526:11052] # [f for f in lst_data if 'albedo' in f]
        lst_shading = [f for f in lst_data][11052:] # [f for f in lst_data if 'shading' in f]
        
        lst_input.sort()
        #lst_mask.sort()
        lst_albedo.sort()
        lst_shading.sort()

        self.lst_input = lst_input
        #self.lst_mask = lst_mask
        self.lst_albedo = lst_albedo
        self.lst_shading = lst_shading

    def __len__(self):
        #assert len(self.lst_input) == len(self.lst_mask)
        assert len(self.lst_input) == len(self.lst_albedo)
        assert len(self.lst_input) == len(self.lst_shading)

        return len(self.lst_input)
	

    def __getitem__(self, index):
        inputs = np.asarray(Image.open(os.path.join(self.data_dir, self.lst_input[index])))
        #mask = np.load(os.path.join(self.data_dir, self.lst_mask[index]))
        albedos = np.asarray(Image.open(os.path.join(self.data_dir, self.lst_albedo[index])))
        shadings = np.asarray(Image.open(os.path.join(self.data_dir, self.lst_shading[index])))
        
        inputs = (inputs/255.0).astype(np.float32)
        albedos = (albedos/255.0).astype(np.float32)
        shadings = (shadings/255.0).astype(np.float32)

        if self.transform:				
            inputs = self.transform(inputs)
            albedos = self.transform(albedos)
            shadings = self.transform(shadings)


        data = {'input':inputs, 
        #'mask':masks, 
        'albedo':albedos, 
        'shading':shadings}

        return data