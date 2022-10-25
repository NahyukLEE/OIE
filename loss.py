import numpy as np
import torch
import torch.nn as nn
from torchmetrics.functional import image_gradients

def recon_loss(mask, gt_i, pred_a, pred_s):
    gt_i = mask * torch.log1p(gt_i) # hadamard product
    pred_a = mask * torch.log1p(pred_a)
    pred_s = mask * torch.log1p(pred_s)
    
    mse_loss = nn.MSELoss(reduction = 'mean')
    loss = mse_loss(gt_i, (pred_a + pred_s))
    
    return loss

def gradient_loss(mask, gt, output):
    
    gt = mask * torch.log1p(gt) # hadamard product
    pred = mask * torch.log1p(output)

    gt_dy, gt_dx = image_gradients(gt)
    pred_dy, pred_dx = image_gradients(output)
    
    l1_loss = nn.L1Loss(reduction = 'mean')
    loss = l1_loss(gt_dy, pred_dy) + l1_loss(gt_dx, pred_dx)

    return loss

def scale_invariant_loss(mask, gt, output):

    gt = mask * torch.log1p(gt) # hadamard product
    pred = mask * torch.log1p(output)

    mse_loss = nn.MSELoss(reduction = 'mean')
    loss = mse_loss(gt, pred)

    return loss

def intrinsic_loss(mask, pred_a, pred_s, gt_a, gt_s, gt_i):

    alpha = 0.5
    albedo_loss = alpha * gradient_loss(mask, gt_a, pred_a) + (1 - alpha) * scale_invariant_loss(mask, gt_a, pred_a)
    shading_loss = alpha * gradient_loss(mask, gt_s, pred_s) + (1 - alpha) * scale_invariant_loss(mask, gt_s, pred_s)
    reconstruction_loss = recon_loss(mask, gt_i, pred_a, pred_s)
    
    w1 = 0.3 # weight for albedo loss
    w2 = 0.3 # weight for shading loss
    loss = w1 * albedo_loss +  w2 * shading_loss + (1-w1-w2) * reconstruction_loss

    return loss

def light_loss(pred_dis, pred_prrs, label_dis, label_prrs, beta=0.1):
    '''
    Implementation heavily borrowed from 
    - https://github.com/PeterZhouSZ/dashcam-illumination-estimation

    '''
    sun_crit = nn.KLDivLoss()
    prr_crit = nn.MSELoss()
    
    sun_loss, prr_loss = sun_crit(pred_dis, label_dis), prr_crit(pred_prrs, label_prrs)
    loss = sun_loss + beta * prr_loss

    return loss

def combine_loss(id_loss, le_loss, alpha=0.5):
    return id_loss + alpha * le_loss