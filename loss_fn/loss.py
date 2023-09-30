from builtins import breakpoint
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from typing import Union



def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """    

    
    
    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + 
            torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    focal_loss = alpha * loss
    
    #focal_loss = torch.sum(weighted_loss)

    #focal_loss /= torch.sum(labels)

    
   

    
    return focal_loss



def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma, device, p=0.8, q=5, eps=1e-2):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.
    Returns:
      cb_loss: A float tensor representing class balanced loss
    """


    effective_num = 1.0 - np.power(beta, samples_per_cls)
    samples_per_cls = torch.tensor(samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = F.one_hot(labels, no_of_classes).float().to(device)
    weights = torch.tensor(weights).float().to(device)
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1,no_of_classes)

    '''
    seesaw_weights = logits.new_ones(labels_one_hot.size()) 

    # mitigation factor
    if p > 0:       
        sample_ratio_matrix = samples_per_cls[None, :].clamp(min=1) / samples_per_cls[:, None].clamp(min=1)
        index = (sample_ratio_matrix < 1.0).float()
        sample_weights = sample_ratio_matrix.pow(p) * index + (1 - index)
        mitigation_factor = sample_weights[labels.long(), :]        
        seesaw_weights = seesaw_weights * (mitigation_factor.to(seesaw_weights.device))

    # compensation factor
    if q > 0:
        scores = F.softmax(logits.detach(), dim=1)
        self_scores = scores[
            torch.arange(0, len(scores)).to(scores.device).long(),
            labels.long()]
        score_matrix = scores / self_scores[:, None].clamp(min=eps)
        index = (score_matrix > 1.0).float()
        compensation_factor = score_matrix.pow(q) * index + (1 - index)
        seesaw_weights = seesaw_weights * (compensation_factor.to(seesaw_weights.device))

    logits = logits + (seesaw_weights.log().to(labels_one_hot.device) * (1 - labels_one_hot))
    '''
   
    if loss_type == "focal":
        p = torch.sigmoid(logits)
       
        pt = labels_one_hot * p + (1 - labels_one_hot) * (1 - p)
        cb_loss = focal_loss(labels_one_hot, pt, weights, gamma)
        cb_loss = cb_loss + 2.0 * torch.pow(1-pt, gamma+1)
        cb_loss = torch.mean(cb_loss)
        
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, pos_weight = weights) 

    elif loss_type == "softmax":
        pred = logits.softmax(dim = 1)
        cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
      
    return cb_loss