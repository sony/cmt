import math
import numpy as np

import torch
import torch.nn as nn
from torch_utils import persistence
from torch_utils import distributed as dist

#----------------------------------------------------------------------------
# Loss function proposed in the blog "Consistency Models Made Easy"

@persistence.persistent_class
class ECMLoss:
    def __init__(self, P_mean=-1.1, P_std=2.0, sigma_data=0.5, q=2, c=0.0, k=8.0, b=1.0, cut=4.0, adj='sigmoid'):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        
        if adj == 'const':
            self.t_to_r = self.t_to_r_const
        elif adj == 'sigmoid':
            self.t_to_r = self.t_to_r_sigmoid
        else:
            raise ValueError(f'Unknow schedule type {adj}!')

        self.q = q
        self.stage = 0
        self.ratio = 0.
        
        self.k = k
        self.b = b

        self.c = c
        dist.print0(f'P_mean: {self.P_mean}, P_std: {self.P_std}, q: {self.q}, k {self.k}, b {self.b}, c: {self.c}')

    def update_schedule(self, stage):
        self.stage = stage
        self.ratio = 1 - 1 / self.q ** (stage+1)

    def t_to_r_const(self, t):
        decay = 1 / self.q ** (self.stage+1)
        ratio = 1 - decay
        r = t * ratio
        return torch.clamp(r, min=0)

    def t_to_r_sigmoid(self, t):
        adj = 1 + self.k * torch.sigmoid(-self.b * t)
        decay = 1 / self.q ** (self.stage+1)
        ratio = 1 - decay * adj
        r = t * ratio
        return torch.clamp(r, min=0)

    def __call__(self, net, images, labels=None, augment_pipe=None):
        # t ~ p(t) and r ~ p(r|t, iters) (Mapping fn)
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        t = (rnd_normal * self.P_std + self.P_mean).exp()
        r = self.t_to_r(t)

        # Augmentation if needed
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        
        # Shared noise direction
        eps   = torch.randn_like(y)
        eps_t = eps * t
        eps_r = eps * r
        
        # Shared Dropout Mask
        rng_state = torch.cuda.get_rng_state()
        D_yt = net(y + eps_t, t, labels, augment_labels=augment_labels)
        
        if r.max() > 0:
            torch.cuda.set_rng_state(rng_state)
            with torch.no_grad():
                D_yr = net(y + eps_r, r, labels, augment_labels=augment_labels)
            
            mask = r > 0
            D_yr = torch.nan_to_num(D_yr)
            D_yr = mask * D_yr + (~mask) * y
        else:
            D_yr = y

        # L2 Loss
        loss = (D_yt - D_yr) ** 2
        loss = torch.sum(loss.reshape(loss.shape[0], -1), dim=-1)
        
        # Producing Adaptive Weighting (p=0.5) through Huber Loss
        if self.c > 0:
            loss = torch.sqrt(loss + self.c ** 2) - self.c
        else:
            loss = torch.sqrt(loss)
        
        # Weighting fn
        return loss / (t - r).flatten()


@persistence.persistent_class
class DPMSolverLoss:
    def __init__(self, NFE, loss_metric):
        from training import dpm_solver
        ns = dpm_solver.NoiseScheduleEDM()
        self.dpm_solver = dpm_solver.DPM_Solver(ns, algorithm_type="dpmsolver++")

        import lpips
        self.lpips_loss_fn = lpips.LPIPS(net='vgg', lpips=True).cuda()

        self.NFE = NFE
        self.loss_metric = loss_metric
        dist.print0("DPMSolverLoss steps=" + str(self.NFE) + " Metric=" + loss_metric)

    def __call__(self, net, net_ref, images, labels=None, augment_pipe=None):
        mini_bs = images.shape[0] // self.NFE
        images = images[:mini_bs]
        if labels is not None: labels = labels[:mini_bs]

        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        T = torch.zeros([images.shape[0], 1, 1, 1], device=images.device) + 80
        t_0 = torch.zeros([images.shape[0], 1, 1, 1], device=images.device) + 0.002
        
        timesteps = self.dpm_solver.get_time_steps(skip_type='logSNR', t_T=T, t_0=t_0, N=self.NFE, device='cuda') # batch, 1, 1, (NFE + 1)
        timesteps = timesteps[:, 0, 0, :-1] # batch, NFE
        timesteps = timesteps.T.reshape(-1, 1, 1, 1) # batch * NFE, 1, 1, 1

        # noise
        eps   = torch.randn_like(y)
        eps_T = eps * T

        s, t = T, t_0
        x = y + eps_T
        with torch.no_grad():
            D_y_ref, Traj = self.dpm_solver.sample(
                net_ref, 
                x, 
                steps=self.NFE, 
                t_start=T, 
                t_end=t_0, 
                method='multistep', 
                return_intermediate=True
            ) # [batch, img_shape], [batch, img_shape] * (NFE + 1)
        # print(D_y_ref.shape, len(Traj), Traj[0].shape)
        D_y_ref = torch.tile(D_y_ref, (self.NFE, 1, 1, 1)) # [batch * NFE, img_shape]
        D_y_ref = torch.nan_to_num(D_y_ref)
        Traj = torch.cat(Traj[:-1], dim=0) # [batch * NFE, img_shape]
        Traj = torch.nan_to_num(Traj) # [batch * NFE, img_shape]

        # print(Traj.shape, timesteps.shape)

        D_yt = net(Traj, timesteps, labels, augment_labels=augment_labels) # [batch * NFE, img_shape]

        # print(D_yt.shape, D_y_ref.shape)

        if self.loss_metric == 'LPIPS':
            loss = self.lpips_loss_fn(D_yt, D_y_ref)
        elif self.loss_metric == "MSE":
            loss = (D_yt - D_y_ref) ** 2
            loss = torch.sum(loss.reshape(loss.shape[0], -1), dim=-1)
        elif self.loss_metric == "MAE":
            loss = (D_yt - D_y_ref) ** 2
            loss = torch.sum(loss.reshape(loss.shape[0], -1), dim=-1)
            loss = torch.sqrt(loss)

        loss = loss.reshape(loss.shape[0],)

        return loss

