import os 
import torch
from loss import *
from utils import *

from models.networks import MultiOutputUNet
from dataset import OutdoorIlluminationDataset
from torch.utils.data import DataLoader
from torchvision import transforms

# configs
learning_rate = 1e-3
batch_size = 4
start_epoch = 0
num_epoch = 100

data_dir = './datasets'
ckpt_dir = './checkpoint'
log_dir  = './log'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data = OutdoorIlluminationDataset(data_dir=os.path.join(data_dir, 'train'))
val_data = OutdoorIlluminationDataset(data_dir=os.path.join(data_dir, 'val'))
#test_data = OutdoorIlluminationDataset(data_dir=os.path.join(data_dir, 'test'))

train_loader = DataLoader(train_data, batch_size = batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size = batch_size, shuffle=True)
#test_loader = DataLoader(test_data, batch_size = batch_size, shuffle=True)

model = MultiOutputUNet().to(device)
optim = torch.optim.Adam(model.parameters(), lr = learning_rate) 


for epoch in range(start_epoch+1,num_epoch +1):
    model.train()
    loss_arr = []

    for batch, data in enumerate(train_loader,1):
        # forward
        inputs = data['input'].to(device)
        gt_a = data['albedo'].to(device)
        gt_s = data['shading'].to(device)

        _, pred_a, pred_s = model(inputs) 

        # backward
        optim.zero_grad()
        #TODO: mask
        loss = intrinsic_loss(mask, pred_a, pred_s, gt_a, gt_s, inputs)
        loss.backward()
        optim.step()

        # save loss
        loss_arr += [loss.item()]
    
    # validation
    with torch.no_grad():
        model.eval()
        loss_arr = []

        for batch, data in enumerate(val_loader,1):
            # forward
            inputs = data['input'].to(device)
            gt_a = data['albedo'].to(device)
            gt_s = data['shading'].to(device)

            # calculate loss
            loss = intrinsic_loss(mask, pred_a, pred_s, gt_a, gt_s, inputs)
            loss_arr += [loss.item()]
            print('valid : epoch %04d / %04d | Batch %04d \ %04d | Loss %04d'%(epoch,num_epoch,batch,num_val_for_epoch,np.mean(loss_arr)))

        # save model
        if epoch % 100 == 0:
            save(ckpt_dir=ckpt_dir, net = model, optim = optim, epoch = epoch)
