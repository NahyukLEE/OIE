import numpy as np
import torch.nn as nn

def recon_loss(mask, gt_i, pred_a, pred_s):

    gt_i = np.multiply(mask, np.log(gt_i)) # hadamard product
    pred_a = np.multiply(mask, np.log(pred_a))
    pred_s = np.multiply(mask, np.log(pred_s))
    
    mse_loss = nn.MSELoss(reduction = 'sum')
    loss = mse_loss(gt_i, (pred_a + pred_s))

    return loss

def gradient_loss(mask, gt, output):

    gt = np.multiply(mask, np.log(gt)) # hadamard product
    pred = np.multiply(mask, np.log(output))

    l1_loss = nn.L1Loss(reduction = 'sum')
    loss = l1_loss(np.gradient(gt), np.gradient(pred))

    return loss

def scale_invariant_loss(mask, gt, output):

    gt = np.multiply(mask, np.log(gt)) # hadamard product
    pred = np.multiply(mask, np.log(output))

    mse_loss = nn.MSELoss(reduction = 'sum')
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
    sun_crit = nn.KLDivLoss()
    prr_crit = nn.MSELoss()
    
    sun_loss, prr_loss = sun_crit(pred_dis, label_dis), prr_crit(pred_prrs, label_prrs)
    loss = sun_loss + beta * prr_loss
    return loss

def combine_loss(id_loss, le_loss, alpha=0.5):
    return id_loss + alpha * le_loss