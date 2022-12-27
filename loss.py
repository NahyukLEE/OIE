import numpy as np
import torch
import torch.nn as nn
from torchmetrics.functional import image_gradients

def recon_loss(mask, gt_i, pred_a, pred_s):
    
    gt_i = mask * torch.log1p(gt_i) # hadamard product
    recon = mask * torch.log1p(pred_a) * torch.log1p(pred_s)
    
    mse_loss = nn.MSELoss(reduction = 'mean')
    loss = mse_loss(gt_i, recon)
    
    return loss

def gradient_loss(mask, gt, pred):
    
    log_gt = torch.log1p(gt) # hadamard product
    log_pred = torch.log1p(mask * pred)

    gt_dy, gt_dx = image_gradients(log_gt)
    pred_dy, pred_dx = image_gradients(log_pred)
    
    l1_loss = nn.L1Loss(reduction = 'mean')
    loss = l1_loss(gt_dy, pred_dy) + l1_loss(gt_dx, pred_dx)

    return loss

def scale_invariant_loss(mask, gt, pred):

    log_gt = torch.log1p(gt) # hadamard product
    log_pred = torch.log1p(mask * pred)

    mse_loss = nn.MSELoss(reduction = 'mean')
    loss = mse_loss(log_gt, log_pred)

    return loss

def intrinsic_loss(mask, pred_a, pred_s, gt_a, gt_s, gt_i):

    alpha = 0.2
    albedo_loss = alpha * gradient_loss(mask, gt_a, pred_a) + (1 - alpha) * scale_invariant_loss(mask, gt_a, pred_a)
    shading_loss = alpha * gradient_loss(mask, gt_s, pred_s) + (1 - alpha) * scale_invariant_loss(mask, gt_s, pred_s)
    reconstruction_loss = recon_loss(mask, gt_i, pred_a, pred_s)
    
    w1 = 0.4 # weight for albedo loss
    w2 = 0.4 # weight for shading loss
    w3 = 0.2 # weight for reconstruction loss
    loss = (w1 * albedo_loss +  w2 * shading_loss + w3 * reconstruction_loss)
    return (loss, albedo_loss, shading_loss, reconstruction_loss)

def intrinsic_loss_only(mask, pred_a, gt_a, gt_i):
    '''
    Implementation for ablation study
    '''
    alpha = 0.2
    albedo_loss = alpha * gradient_loss(mask, gt_a, pred_a) + (1 - alpha) * scale_invariant_loss(mask, gt_a, pred_a)
     
    loss = albedo_loss
    return (loss, albedo_loss)


def light_loss(pred_dis, label_dis, beta=0.1):
    '''
    Implementation heavily borrowed from 
    - https://github.com/PeterZhouSZ/dashcam-illumination-estimation

    '''
    sun_crit = nn.KLDivLoss(reduction='batchmean')
    prr_crit = nn.MSELoss()
    sun_loss = sun_crit(pred_dis, label_dis)
    
    #prr_loss = prr_crit(pred_prrs, label_prrs)
    #loss = sun_loss + beta * prr_loss

    return sun_loss #loss

def combine_loss(id_loss, le_loss, alpha=0.5):
    return id_loss + le_loss