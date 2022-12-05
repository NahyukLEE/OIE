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
wandb.init(entity="nahyuklee",  
        project="default",
        name= now.strftime('%Y-%m-%d %H:%M:%S'))

# configs
learning_rate = 1e-3
batch_size = 16
start_epoch = 0
num_epoch = 20

data_dir = './fov_dataset'
ckpt_dir = './checkpoint'
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

optim = torch.optim.Adam(id_model.parameters(), lr = learning_rate) 
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optim,
                                        lr_lambda=lambda epoch: 0.95 ** epoch)
num_data_train = len(train_data)
num_data_val = len(val_data)
print('train data:', num_data_train)
print('validation data:', num_data_val)

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
        id_loss, al, sl, rl = intrinsic_loss(mask, pred_a, pred_s, gt_a, gt_s, inputs)
        #le_loss = light_loss(pred_dis, pred_prrs, gt_dis, gt_prrs)
        loss = id_loss #loss = combine_loss(id_loss, le_loss, 0.5)

        # weight update
        loss.backward()
        optim.step()

        # save loss
        loss_arr += [loss.item()]
        
        i = to_pil_image(inputs[0])
        a = to_pil_image(mask[0] * pred_a[0])
        s = to_pil_image(mask[0] * pred_s[0])
        i.save(f'./results/{epoch}_input.jpg')
        a.save(f'./results/{epoch}_albedo.jpg')
        s.save(f'./results/{epoch}_shading.jpg')

        wandb.log({'train intrinsic loss': id_loss,
                 'train albedo loss': al,
                 'train shading loss': sl,
                 'train recon. loss': rl,
                 'learning rate': optim.param_groups[0]['lr'],
                 })
    
    scheduler.step()
    print("TRAIN: EPOCH %04d / %04d | LOSS %.4f" %(epoch, num_epoch, np.mean(loss_arr)))
        

    # validation
    with torch.no_grad():
        id_model.eval()
        loss_arr = []

        for batch, data in enumerate(val_loader,1):
            inputs = data['input'].to(device)
            mask = data['mask'].to(device)
            gt_a = data['albedo'].to(device)
            gt_s = data['shading'].to(device)
            #gt_dis = data['distribution'].to(device)
            #gt_prrs = data['params'].to(device)

            # forward pass
            feature, pred_a, pred_s = id_model(inputs) 
            

            #pred_dis, pred_prrs = le_model(feature)
            
            # calculate loss
            id_loss, al, sl, rl = intrinsic_loss(mask, pred_a, pred_s, gt_a, gt_s, inputs)
            #le_loss = light_loss(pred_dis, pred_prrs, gt_dis, gt_prrs)
            loss = id_loss #combine_loss(id_loss, le_loss, 0.5)

            loss_arr += [loss.item()]

            wandb.log({'valid intrinsic loss': id_loss,
                            'valid albedo loss': al,
                            'valid shading loss': sl,
                            'valid recon. loss': rl,
                            })

        print("VALID: EPOCH %04d / %04d | LOSS %.4f" %
                (epoch, num_epoch, np.mean(loss_arr)))

        # save model
        if epoch % 100 == 0:
            save(ckpt_dir=ckpt_dir, net = id_model, optim = optim, epoch = epoch)
