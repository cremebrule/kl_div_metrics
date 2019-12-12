import numpy as np
import torch
from torch.nn import functional as F

'''
A set of utility functions that are used by all other scripts / classes
'''

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def compute_KL(x, model):
    out = model.classifier(x)
    y_prob = F.softmax(out, dim=-1).float()
    qm, qv = model.encoder(x, y_prob)
    kl_z_all = y_prob * kl_normal(qm, qv, model.z_prior[0],
                                    model.z_prior[1])  # kl_z_all shape = [batch_size * y_dim]
    kl_z = torch.sum(kl_z_all)  # scalar

    return kl_z

def kl_normal(qm, qv, pm, pv):
    element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
    kl = element_wise.sum(-1)
    return kl