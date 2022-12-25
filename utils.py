import os
import torch

import os
import sys
import cv2
import numpy as np
from math import *

# https://github.com/fuenwang/Equirec2Perspec/blob/master/Equirec2Perspec.py
def xyz2lonlat(xyz):
    atan2 = np.arctan2
    asin = np.arcsin

    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    xyz_norm = xyz / norm
    x = xyz_norm[..., 0:1]
    y = xyz_norm[..., 1:2]
    z = xyz_norm[..., 2:]

    lon = atan2(x, z)
    lat = asin(y)
    lst = [lon, lat]

    out = np.concatenate(lst, axis=-1)
    return out

def lonlat2XY(lonlat, shape):
    X = (lonlat[..., 0:1] / (2 * np.pi) + 0.5) * (shape[1] - 1)
    Y = (lonlat[..., 1:] / (np.pi) + 0.5) * (shape[0] - 1)
    lst = [X, Y]
    out = np.concatenate(lst, axis=-1)

    return out 

class Equirectangular:
    def __init__(self, img_array):
        self._img = img_array
        [self._height, self._width, _] = self._img.shape
        #cp = self._img.copy()  
        #w = self._width
        #self._img[:, :w/8, :] = cp[:, 7*w/8:, :]
        #self._img[:, w/8:, :] = cp[:, :7*w/8, :]
    

    def GetPerspective(self, FOV, THETA, PHI, height, width):
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #

        f = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        K = np.array([
                [f, 0, cx],
                [0, f, cy],
                [0, 0,  1],
            ], np.float32)
        K_inv = np.linalg.inv(K)
        
        x = np.arange(width)
        y = np.arange(height)
        x, y = np.meshgrid(x, y)
        z = np.ones_like(x)
        xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
        xyz = xyz @ K_inv.T

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        x_axis = np.array([1.0, 0.0, 0.0], np.float32)
        R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
        R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
        R = R2 @ R1
        xyz = xyz @ R.T
        lonlat = xyz2lonlat(xyz) 
        XY = lonlat2XY(lonlat, shape=self._img.shape).astype(np.float32)
        persp = cv2.remap(self._img, XY[..., 0], XY[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)

        return persp

def sphere2world(theta, phi):
    x = cos(radians(phi)) * sin(radians(theta))
    y = sin(radians(phi))
    z = cos(radians(phi)) * cos(radians(theta))
    return np.asarray([x, y, z])

def bin2Sphere(i):
    phi = (floor(i/32)) * (90/8.0) + (90/16.0)
    theta = ((i+1) - floor(i/32) * 32 - 1) * (360.0/32.0) + (360.0/64.0) - 180.0
    return np.array([theta, phi])

def vMF(SP, kappa=80.0):	
	'''
		discrete the sky into 256 bins and model the sky probability distirbution. (von Mises-Fisher)
	'''
	sp_vec = sphere2world(SP[0], SP[1])
	pdf = np.zeros(256)
	for i in range(256):
		sp = bin2Sphere(i)
		vec = sphere2world(sp[0], sp[1])
		pdf[i] = exp(np.dot(vec, sp_vec) * kappa)
	return pdf/np.sum(pdf)

def save(ckpt_dir,model,optim,epoch,id):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'epoch': epoch,
                'model':model.state_dict(),
                'optim':optim.state_dict()},'%s/%smodel_epoch%d.pth'%(ckpt_dir,id,epoch))

