import os 
import torch

from models.networks import MultiOutputUNet
from dataset import OutdoorIlluminationDataset

# configs
learning_rate = 1e-3
batch_size = 4
num_epoch = 100

data_dir = './datasets'
ckpt_dir = './checkpoint'
log_dir  = './log'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data = OutdoorIlluminationDataset(data_dir=os.path.join(data_dir, 'train'))
val_data = OutdoorIlluminationDataset(data_dir=os.path.join(data_dir, 'val'))
test_data = OutdoorIlluminationDataset(data_dir=os.path.join(data_dir, 'test'))

