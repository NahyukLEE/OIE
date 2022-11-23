import os 
import torch
from loss import *
from utils import *

from models.networks import MultiOutputUNet, FCRegressor
from dataset import OutdoorIlluminationDataset
from torch.utils.data import DataLoader
from torchvision import transforms

# configs
learning_rate = 1e-3
batch_size = 1
start_epoch = 0
num_epoch = 100
loss_weight = 0.5

data_dir = './sample_data'
ckpt_dir = './checkpoint'
log_dir  = './log'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.ToTensor()])

train_data = OutdoorIlluminationDataset(data_dir=os.path.join(data_dir), transform=transform)
val_data = OutdoorIlluminationDataset(data_dir=os.path.join(data_dir), transform=transform)
#test_data = OutdoorIlluminationDataset(data_dir=os.path.join(data_dir, 'test'), transform)

train_loader = DataLoader(train_data, batch_size = batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size = batch_size, shuffle=True)
#test_loader = DataLoader(test_data, batch_size = batch_size, shuffle=False)

id_model = MultiOutputUNet().to(device)
le_model = FCRegressor().to(device)

optim = torch.optim.Adam(id_model.parameters(), lr = learning_rate) 

num_data_train = len(train_data)
num_data_val = len(val_data)

num_batch_train = np.ceil(num_data_train / batch_size)
num_batch_val = np.ceil(num_data_val / batch_size)

for epoch in range(start_epoch+1,num_epoch +1):
    id_model.train()
    le_model.train()
    loss_arr = []

    for batch, data in enumerate(train_loader,1):
        inputs = data['input'].to(device)
        mask = data['mask'].to(device)
        gt_a = data['albedo'].to(device)
        gt_s = data['shading'].to(device)
        #gt_dis = data['distribution'].to(device)
        #gt_prrs = data['params'].to(device)

        # forward pass
        feature, pred_a, pred_s = id_model(inputs) 
        #pred_dis, pred_prrs = le_model(feature)

        # backward
        optim.zero_grad()

        # calculate loss
        id_loss = intrinsic_loss(mask, pred_a, pred_s, gt_a, gt_s, inputs)
        #le_loss = light_loss(pred_dis, pred_prrs, gt_dis, gt_prrs)
        loss = id_loss #loss = combine_loss(id_loss, le_loss, 0.5)

        # weight update
        loss.backward()
        optim.step()

        # save loss
        loss_arr += [loss.item()]

        print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                  (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))
        

    # validation
    with torch.no_grad():
        id_model.eval()
        loss_arr = []

        for batch, data in enumerate(val_loader,1):
            inputs = data['input'].to(device)
            #mask = data['mask'].to(device)
            gt_a = data['albedo'].to(device)
            gt_s = data['shading'].to(device)
            #gt_dis = data['distribution'].to(device)
            #gt_prrs = data['params'].to(device)

            # forward pass
            feature, pred_a, pred_s = id_model(inputs) 
            #pred_dis, pred_prrs = le_model(feature)
            
            # calculate loss
            id_loss = intrinsic_loss(mask, pred_a, pred_s, gt_a, gt_s, inputs)
            #le_loss = light_loss(pred_dis, pred_prrs, gt_dis, gt_prrs)
            loss = id_loss #combine_loss(id_loss, le_loss, 0.5)

            loss_arr += [loss.item()]

            print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                  (epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)))

        # save model
        if epoch % 100 == 0:
            save(ckpt_dir=ckpt_dir, net = id_model, optim = optim, epoch = epoch)
