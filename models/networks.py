import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.DANet import PositionAttentionModule, ChannelAttentionModule, DAAttention


class MultiOutputUNet(nn.Module):
    '''
    Multi Output UNet: base network
    - input: fov rgb image
    - outputs: attention, albedo, shading 
    '''
    def __init__(self):
        super(MultiOutputUNet, self).__init__()

        # Conv-Batchnorm-Relu
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True): 
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr
        
        # Self Attention Modules
        self.attention = DAAttention(in_channels=1024)

        # Contracting Path
        self.enc1_1 = CBR2d(in_channels=3, out_channels=64) # rgb input
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)

        # Expansive Path for albedo
        self.dec5_1_a = CBR2d(in_channels=1024, out_channels=512)
        
        self.unpool4_a = nn.ConvTranspose2d(in_channels=512, out_channels=512,
        kernel_size=2, stride=2, padding=0, bias=True)
        self.dec4_2_a = CBR2d(in_channels=1024, out_channels=512)
        self.dec4_1_a = CBR2d(in_channels=512, out_channels=256)

        self.unpool3_a = nn.ConvTranspose2d(in_channels=256, out_channels=256,
        kernel_size=2, stride=2, padding=0, bias=True)
        self.dec3_2_a = CBR2d(in_channels=512, out_channels=256)
        self.dec3_1_a = CBR2d(in_channels=256, out_channels=128)

        self.unpool2_a = nn.ConvTranspose2d(in_channels=128, out_channels=128,
        kernel_size=2, stride=2, padding=0, bias=True)
        self.dec2_2_a = CBR2d(in_channels=256, out_channels=128)
        self.dec2_1_a = CBR2d(in_channels=128, out_channels=64)

        self.unpool1_a = nn.ConvTranspose2d(in_channels=64, out_channels=64,
        kernel_size=2, stride=2, padding=0, bias=True)
        self.dec1_2_a = CBR2d(in_channels=128, out_channels=64)
        self.dec1_1_a = CBR2d(in_channels=64, out_channels=64)

        # Expansive Path for shading
        self.dec5_1_s = CBR2d(in_channels=1024, out_channels=512)
        
        self.unpool4_s = nn.ConvTranspose2d(in_channels=512, out_channels=512,
        kernel_size=2, stride=2, padding=0, bias=True)
        self.dec4_2_s = CBR2d(in_channels=1024, out_channels=512)
        self.dec4_1_s = CBR2d(in_channels=512, out_channels=256)

        self.unpool3_s = nn.ConvTranspose2d(in_channels=256, out_channels=256,
        kernel_size=2, stride=2, padding=0, bias=True)
        self.dec3_2_s = CBR2d(in_channels=512, out_channels=256)
        self.dec3_1_s = CBR2d(in_channels=256, out_channels=128)

        self.unpool2_s = nn.ConvTranspose2d(in_channels=128, out_channels=128,
        kernel_size=2, stride=2, padding=0, bias=True)
        self.dec2_2_s = CBR2d(in_channels=256, out_channels=128)
        self.dec2_1_s = CBR2d(in_channels=128, out_channels=64)

        self.unpool1_s = nn.ConvTranspose2d(in_channels=64, out_channels=64,
        kernel_size=2, stride=2, padding=0, bias=True)
        self.dec1_2_s = CBR2d(in_channels=128, out_channels=64)
        self.dec1_1_s = CBR2d(in_channels=64, out_channels=64)

        self.fc_a = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc_s = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1, stride=1, padding=0, bias=True)

        self.relu = nn.ReLU()

    def forward(self, x):
        # Encoding
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)
        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)
        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)
        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)
        enc5_1 = self.enc5_1(pool4)

        # Attention
        attention = self.attention(enc5_1)

        # Decoding (albedo)
        dec5_1_a = self.dec5_1_a(attention)
        unpool4_a = self.unpool4_a(dec5_1_a)
        cat4_a = torch.cat((unpool4_a, enc4_2), dim=1) # dim=[0:batch, 1:channel, 2:height, 3:width]
        dec4_2_a = self.dec4_2_a(cat4_a)
        dec4_1_a = self.dec4_1_a(dec4_2_a)
        unpool3_a = self.unpool3_a(dec4_1_a)
        cat3_a = torch.cat((unpool3_a, enc3_2), dim=1) 
        dec3_2_a = self.dec3_2_a(cat3_a)
        dec3_1_a = self.dec3_1_a(dec3_2_a)
        unpool2_a = self.unpool2_a(dec3_1_a)
        cat2_a = torch.cat((unpool2_a, enc2_2), dim=1)
        dec2_2_a = self.dec2_2_a(cat2_a)
        dec2_1_a = self.dec2_1_a(dec2_2_a)
        unpool1_a = self.unpool1_a(dec2_1_a)
        cat1_a = torch.cat((unpool1_a, enc1_2), dim=1)
        dec1_2_a = self.dec1_2_a(cat1_a)
        dec1_1_a = self.dec1_1_a(dec1_2_a)
        albedo = self.relu(self.fc_a(dec1_1_a))

        # Decoding (shading)
        dec5_1_s = self.dec5_1_s(attention)
        unpool4_s = self.unpool4_s(dec5_1_s)
        cat4_s = torch.cat((unpool4_s, enc4_2), dim=1)
        dec4_2_s = self.dec4_2_s(cat4_s)
        dec4_1_s = self.dec4_1_s(dec4_2_s)
        unpool3_s = self.unpool3_s(dec4_1_s)
        cat3_s = torch.cat((unpool3_s, enc3_2), dim=1) 
        dec3_2_s = self.dec3_2_s(cat3_s)
        dec3_1_s = self.dec3_1_s(dec3_2_s)
        unpool2_s = self.unpool2_s(dec3_1_s)
        cat2_s = torch.cat((unpool2_s, enc2_2), dim=1)
        dec2_2_s = self.dec2_2_s(cat2_s)
        dec2_1_s = self.dec2_1_s(dec2_2_s)
        unpool1_s = self.unpool1_s(dec2_1_s)
        cat1_s = torch.cat((unpool1_s, enc1_2), dim=1)
        dec1_2_s = self.dec1_2_s(cat1_s)
        dec1_1_s = self.dec1_1_s(dec1_2_s)
        shading = self.relu(self.fc_s(dec1_1_s))

        return attention, albedo, shading
        
# Fully Connected for Lighting Regression
class FCRegressor(nn.Module):
    '''
    Fully Connected Layer for Lighting Regression
    - input: attention feature vector
    - output: sky distribution, sky & light parameters
    '''
    def __init__(self, in_channels=1024):
        super(FCRegressor, self).__init__()

        self.linear = nn.Linear(in_channels, 512)
        self.fc_bn = nn.BatchNorm1d(2048)

        ''' two heads regression'''
        self.ds_fc = nn.Linear(512, 256)
        self.ds_bn = nn.BatchNorm1d(256)
        self.pr_fc = nn.Linear(512, 4)
        self.pr_bn = nn.BatchNorm1d(4)

    def forward(self, x):
        x = self.linear(x)
        x = F.elu(self.fc_bn(x))

        dist = F.log_softmax(self.ds_bn(self.ds_fc(x)), dim=1)
        param = self.pr_bn(self.pr_fc(x))
        return dist, param