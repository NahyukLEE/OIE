import os 
import torch

from models.model import BaseUNet

# configs
learning_rate = 1e-3
batch_size = 4
num_epoch = 100

data_dir = './datasets'
ckpt_dir = './checkpoint'
log_dir  = './log'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
