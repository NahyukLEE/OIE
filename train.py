import os 
import torch
from loss import *
from utils import *

from models.networks import MultiOutputUNet, FCRegressor
from dataset import OutdoorIlluminationDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

import wandb
import time

# wandb
now = time

wandb.init(entity="cvar-cau",  
        project="OIEID",
        name= now.strftime('%Y-%m-%d %H:%M:%S'))


# configs
learning_rate = 1e-3
batch_size = 16
start_epoch = 0
num_epoch = 30

data_dir = './fov_dataset_new'
ckpt_dir = './checkpoints'
log_dir  = './log'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.Resize((240,320)),
                                transforms.ToTensor(),
                                ])

train_data = OutdoorIlluminationDataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)
val_data = OutdoorIlluminationDataset(data_dir=os.path.join(data_dir, 'val'), transform=transform)
test_data = OutdoorIlluminationDataset(data_dir=os.path.join(data_dir, 'test'), transform=transform)

train_loader = DataLoader(train_data, batch_size = batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size = batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size = batch_size, shuffle=False)

id_model = MultiOutputUNet().to(device)
le_model = FCRegressor().to(device)

optim_id = torch.optim.Adam(id_model.parameters(), lr = learning_rate) 
optim_le = torch.optim.Adam(le_model.parameters(), lr = learning_rate) 

scheduler_id = torch.optim.lr_scheduler.LambdaLR(optimizer=optim_id,
                                        lr_lambda=lambda epoch: 0.95 ** epoch)
scheduler_le = torch.optim.lr_scheduler.LambdaLR(optimizer=optim_le,
                                        lr_lambda=lambda epoch: 0.95 ** epoch)

num_data_train = len(train_data)
num_data_val = len(val_data)
num_data_test = len(test_data)
print('train data:', num_data_train)
print('validation data:', num_data_val)
print('test data:', num_data_test)

num_batch_train = np.ceil(num_data_train / batch_size)
num_batch_val = np.ceil(num_data_val / batch_size)
num_data_test = np.ceil(num_data_test / batch_size)

for epoch in range(start_epoch+1,num_epoch +1):
    id_model.train()
    le_model.train()

    loss_arr = []

    for batch, data in enumerate(train_loader,1):
        inputs = data['input'].to(device)
        mask = data['mask'].to(device)
        gt_a = data['albedo'].to(device)
        gt_s = data['shading'].to(device)
        gt_dis = data['dis'].to(device)

        # forward pass
        feature, pred_a, pred_s = id_model(inputs) 
        pred_dis = le_model(feature)
        
        # backward
        optim_id.zero_grad()
        optim_le.zero_grad()

        # calculate loss
        id_loss, al, sl, rl = intrinsic_loss(mask, pred_a, pred_s, gt_a, gt_s, inputs)
        le_loss = light_loss(pred_dis, gt_dis)
        loss = combine_loss(id_loss, le_loss, 0.5)

        # weight update
        loss.backward()

        optim_id.step()
        optim_le.step()
        
        # save loss
        loss_arr += [loss.item()]

        wandb.log({'train intrinsic loss': id_loss,
                'train light estimation loss': le_loss,
                 'train albedo loss': al,
                 'train shading loss': sl,
                 'train recon. loss': rl,
                 'learning rate': optim_id.param_groups[0]['lr'],
                 })

    scheduler_id.step()
    scheduler_le.step()
    print("TRAIN: EPOCH %04d / %04d | LOSS %.4f" %(epoch, num_epoch, np.mean(loss_arr)))
        

    # validation
    with torch.no_grad():
        id_model.eval()
        le_model.eval()

        loss_arr = []

        for batch, data in enumerate(val_loader,1):
            inputs = data['input'].to(device)
            mask = data['mask'].to(device)
            gt_a = data['albedo'].to(device)
            gt_s = data['shading'].to(device)
            gt_dis = data['dis'].to(device)

            # forward pass
            feature, pred_a, pred_s = id_model(inputs) 
            pred_dis = le_model(feature)
            
            # calculate loss
            id_loss, al, sl, rl = intrinsic_loss(mask, pred_a, pred_s, gt_a, gt_s, inputs)
            le_loss = light_loss(pred_dis, gt_dis)
            loss = combine_loss(id_loss, le_loss, 0.5)

            # save loss
            loss_arr += [loss.item()]

            wandb.log({'valid intrinsic loss': id_loss,
                        'valid light estimation loss': le_loss,
                            'valid albedo loss': al,
                            'valid shading loss': sl,
                            'valid recon. loss': rl,
                            })

        print("VALID: EPOCH %04d / %04d | LOSS %.4f" %
                (epoch, num_epoch, np.mean(loss_arr)))

    # save model
    if epoch % 1 == 0:
        save(ckpt_dir=os.path.join(ckpt_dir,now.strftime('%Y-%m-%d %H:%M:%S')), net = id_model, optim = optim_id, epoch = epoch, id = 'id')
        save(ckpt_dir=os.path.join(ckpt_dir,now.strftime('%Y-%m-%d %H:%M:%S')), net = le_model, optim = optim_le, epoch = epoch, id = 'le')
