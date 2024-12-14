# ---------------------------------------------------------------
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from ECT codebase, which built upon EDM.
#
# Source:
# https://github.com/NVlabs/edm/blob/main/training/loss.py (EDM)
# https://github.com/locuslab/ect/blob/main/training/loss.py (ECT)
#
# The license for these can be found in license/ directory.
# The modifications to this file are subject to the same license.
# ---------------------------------------------------------------

import math
import numpy as np
import torch
import torch.nn as nn
from torch_utils import persistence
from torch_utils import distributed as dist
from torch_utils import misc
from functools import partial
import os
from training.networks import inplace_norm_flag
#----------------------------------------------------------------------------

@torch.no_grad()
def ode_solver(score_net, samples, t, next_t, labels, augment_labels):
    x_t = samples
    denoiser = score_net(x_t, t, labels, augment_labels=augment_labels)
    d = (x_t - denoiser) / t
    samples = x_t + d * (next_t - t)

    return samples

def sigmoid(t):
    return 1 / (1 + (-t).exp())
def gaussian_pdf(x, mu, sigma):
    return (1.0 / (sigma * torch.sqrt(2 * torch.tensor(torch.pi)))) * torch.exp(-0.5 * ((x - mu) / sigma) ** 2)

def delta_sigmoid(t, ratio):
    # Sigmoid delta t proposed by ECT
    dt = t * (1-ratio) * (1 + 8 * sigmoid(-t))
    s = t - dt
    s[s<1e-6] = 1e-6
    dt = t - s
    return dt


@persistence.persistent_class
class ECMLoss(nn.Module):
    def __init__(self, P_mean=-1.1, P_std=2.0, t_lower = 0.002, t_upper = 80, sigma_data=0.5, q=2, c=0.0, k=8.0, b=1.0, cut=4.0, delta='sigmoid', ratio_limit=0.999, sqrt=True, weighting='default', boundary_prob=0.05, tdist='normal', df=2):
        super().__init__()

        self.P_mean = P_mean
        self.P_std = P_std
        self.t_lower = t_lower
        self.t_upper = t_upper # t \in [t_lower, t_upper]
        self.sigma_data = sigma_data
        self.ratio_limit = ratio_limit
        self.sqrt = sqrt
        self.weighting = weighting
        self.delta = delta
        self.boundary_prob = boundary_prob
        self.tdist = tdist # normal or t
        self.df = df # degrees of freedom for t distribution

        self.q = q
        self.stage = 0
        self.ratio = 0.

        self.k = k
        self.b = b
        self.cut = cut

        self.c = c
        dist.print0(
            f'P_mean: {self.P_mean}, P_std: {self.P_std}, q: {self.q}, k {self.k}, b {self.b}, cut {self.cut}, c: {self.c}')

    def update_schedule(self, stage):
        self.stage = stage
        self.ratio = 1 - 1 / self.q ** (stage + 1)
        if self.ratio > self.ratio_limit:
            dist.print0(f"Clipping ratio from {self.ratio} -> {self.ratio_limit}")
            self.ratio = self.ratio_limit
            


    

    def __call__(self, net, images, labels=None, augment_pipe=None, teacher_net=None, t=None):
        delta = delta_sigmoid
        if t is None:
            if self.tdist == 'normal':
                logt = misc.truncated_normal(images.shape[0], mu=self.P_mean, sigma=self.P_std, lower=np.log(self.t_lower), upper=np.log(self.t_upper)) # np.array
            elif self.tdist == 't':
                logt = misc.truncated_t(images.shape[0], mu=self.P_mean, sigma=self.P_std, lower=np.log(self.t_lower), upper=np.log(self.t_upper), df=self.df)
            logt = torch.tensor(logt, device=images.device).view(-1, 1, 1, 1)
            t = logt.exp()
            if hasattr(net, 'transition_t'):
                num_elements_to_mask = int(t.shape[0] * self.boundary_prob)
                indices = torch.randperm(t.shape[0])
                mask_indices = indices[:num_elements_to_mask]
                mask_t = torch.zeros_like(t, dtype=torch.bool)
                mask_t[mask_indices] = True

                t = mask_t * (net.transition_t+1e-8) + ~mask_t * t

        r = t - delta(t, self.ratio)

        # Augmentation
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)

        # Shared noise direction
        eps = torch.randn_like(y)
        y_t = y + eps * t
        if teacher_net is None:
            y_r = y + eps * r
        else:
            y_r = ode_solver(teacher_net, y_t, t, r, labels, augment_labels=augment_labels).detach()

        # Shared Dropout Mask
        rng_state = torch.cuda.get_rng_state()
        D_yt = net(y_t, t, labels, augment_labels=augment_labels)
       
        if r.max() > 0:
            torch.cuda.set_rng_state(rng_state)
            with torch.no_grad():
                token = inplace_norm_flag.set(False)
                D_yr = net(y_r, r, labels, augment_labels=augment_labels)
                inplace_norm_flag.reset(token)
            mask = r > 0
            D_yr = torch.nan_to_num(D_yr)
            D_yr = mask * D_yr + (~mask) * y
        else:
            D_yr = y

        # L2 Loss
        l2_distance = torch.norm(D_yt - D_yr, dim=(1, 2, 3), p=2)

        
        # Huber Loss if needed
        if self.c > 0:
            loss_unweighted = torch.sqrt(l2_distance ** 2 + self.c ** 2) - self.c
        else:
            if self.sqrt:
               loss_unweighted = l2_distance
            else:
               loss_unweighted = l2_distance ** 2
            
        # Weighting fn


        t = t.flatten()
        r = r.flatten()

        if self.weighting == 'default':
            loss = loss_unweighted / delta(t, self.ratio)
        elif self.weighting == 'cout':
            loss = loss_unweighted / misc.get_edm_cout(t, sigma_data = self.sigma_data)
        elif self.weighting == 'cout_sq':
            loss = loss_unweighted / misc.get_edm_cout(t, sigma_data = self.sigma_data) ** 2
        elif self.weighting == 'sqrt':
            loss = loss_unweighted / (t-r)**0.5
        elif self.weighting == 'one':
            loss = loss_unweighted
        else:
            raise NotImplementedError(f"Weighting function {self.weighting} not implemented.")
        
        if hasattr(net, 'transition_t') and torch.any(mask_t):
            loss_boundary = loss[mask_t.flatten()]      
            loss = loss[~mask_t.flatten()]  
            loss_unweighted = loss_unweighted[~mask_t.flatten()]
            t = t[~mask_t.flatten()]
            l2_distance = l2_distance[~mask_t.flatten()]
        else:
            loss_boundary = torch.zeros_like(loss)
        return loss_unweighted, loss, t, l2_distance, loss_boundary
