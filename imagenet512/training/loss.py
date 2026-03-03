import contextlib
import math

import torch
import torch.nn as nn
from torch_utils import persistence
from torch_utils import distributed as dist

from .networks_edm2 import MPConv

#----------------------------------------------------------------------------
# The forced WN can introduce small numerical errors.
# Disable it in the self teacher step.

@contextlib.contextmanager
def disable_forced_wn(module):
    def set_force_wn(m):
        if isinstance(m, MPConv):
            m.force_wn = False

    def reset_force_wn(m):
            if isinstance(m, MPConv):
                m.force_wn = True

    module.apply(set_force_wn)
    try:
        yield
    finally:
        module.apply(reset_force_wn)

#----------------------------------------------------------------------------
# Loss function in "Consistency Models Made Easy"

@persistence.persistent_class
class ECMLoss:
    def __init__(self, 
            P_mean=-1.1, P_std=2.0, sigma_data=0.5, 
            q=4, c=0.0, k=8.0, b=1.0, adj='sigmoid', wt='snrpk'
            ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
       
        if adj == 'const':
            dist.print0('const adj')
            self.t_to_r = self.t_to_r_const
        elif adj == 'sigmoid':
            dist.print0('sigmoid adj')
            self.t_to_r = self.t_to_r_sigmoid
        else:
            raise ValueError(f'Unknow schedule type {adj}!')
        
        if wt == 'snrpk':
            self.wt_fn = self.snrplusk_wt
        else:
            raise ValueError(f'Unknow wt fn type {adj}!')

        self.q = q
        self.stage = 0
        self.ratio = 0.
        
        self.k = k
        self.b = b

        self.c = c
        dist.print0(f'Wt: {wt}, P_mean: {self.P_mean}, P_std: {self.P_std}, q: {self.q}, k {self.k}, b {self.b}, c: {self.c}')

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

    def snrplusk_wt(self, t, r):
        # SNR(t) + k = 1/t**2 + k
        wt = (t ** 2 + self.sigma_data ** 2) / (t * self.sigma_data) ** 2
        return wt

    def __call__(self, net, images, labels=None):
        # t ~ p(t) and r ~ p(r|t, iters) (Mapping fn)
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        t = (rnd_normal * self.P_std + self.P_mean).exp()
        r = self.t_to_r(t)

        # Augmentation
        x_0 = images
        
        # Shared noise direction
        eps = torch.randn_like(x_0)
        x_t = x_0 + eps * t
        x_r = x_0 + eps * r
        
        # Shared Dropout Mask
        rng_state = torch.cuda.get_rng_state()
        fx_t = net(x_t, t, labels)
        
        if r.max() > 0:
            torch.cuda.set_rng_state(rng_state)
            with torch.no_grad():
                with disable_forced_wn(net):
                    fx_r = net(x_r, r, labels)
            
            mask = r > 0
            fx_r = torch.nan_to_num(fx_r)
            fx_r = mask * fx_r + (~mask) * x_0
        else:
            fx_r = x_0

        # L2 Loss
        loss = (fx_t - fx_r) ** 2
        loss = torch.sum(loss.reshape(loss.shape[0], -1), dim=-1)
        
        if self.c > 0:
            loss = torch.sqrt(loss + self.c ** 2) - self.c
        else:
            loss = torch.sqrt(loss)
        
        wt = self.wt_fn(t, r)
        return loss * wt.flatten()


@persistence.persistent_class
class DPMSolverLoss:
    def __init__(self, NFE, loss_metric, c=0.06):
        from training import dpm_solver
        ns = dpm_solver.NoiseScheduleEDM()
        self.dpm_solver = dpm_solver.DPM_Solver(ns, algorithm_type="dpmsolver++")

        self.NFE = NFE
        self.loss_metric = loss_metric
        dist.print0("DPMSolverLoss steps=" + str(self.NFE) + " Metric=" + loss_metric)
        self.c = c

        if self.loss_metric == "ELatentLPIPS":
            from elatentlpips import ELatentLPIPS2
            self.elatentlpips = ELatentLPIPS2(
                pretrained_latent_lpips_path="vgg.pt", 
                aug_type="bgcfnc").to("cuda")

    def __call__(self, net, net_ref, images, labels=None):
        # print(labels)
        # print(labels.shape)
        # tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        # [0., 0., 0.,  ..., 0., 0., 0.],
        # [0., 0., 0.,  ..., 0., 0., 0.],
        # ...,
        # [0., 0., 0.,  ..., 0., 0., 0.],
        # [0., 0., 0.,  ..., 0., 0., 0.],
        # [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0')
        # torch.Size([64, 1000])

        mini_bs = images.shape[0] // self.NFE
        images = images[:mini_bs]
        if labels is not None: labels = labels[:mini_bs]
        #print(labels.shape)

        T = torch.zeros([images.shape[0], 1, 1, 1], device=images.device) + 80
        t_0 = torch.zeros([images.shape[0], 1, 1, 1], device=images.device) + 0.002
        
        timesteps = self.dpm_solver.get_time_steps(skip_type='logSNR', t_T=T, t_0=t_0, N=self.NFE, device='cuda') # batch, 1, 1, (NFE + 1)
        timesteps = timesteps[:, 0, 0, :-1] # batch, NFE
        timesteps = timesteps.T.reshape(-1, 1, 1, 1) # batch * NFE, 1, 1, 1

        # noise
        eps   = torch.randn_like(images)
        eps_T = eps * T

        x = images + eps_T

        net_ref_fn = lambda x, t: net_ref(x, t, labels) 
        with torch.no_grad():
            D_y_ref, Traj = self.dpm_solver.sample(
                net_ref_fn, 
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

        labels = torch.tile(labels, (self.NFE, 1))

        # print(Traj.shape, timesteps.shape)

        D_yt = net(Traj, timesteps, labels) # [batch * NFE, img_shape]

        # print(D_yt.shape, D_y_ref.shape)

        if self.loss_metric == "MSE":
            loss = (D_yt - D_y_ref) ** 2
            loss = torch.sum(loss.reshape(loss.shape[0], -1), dim=-1)
        elif self.loss_metric == "MAE":
            loss = (D_yt - D_y_ref) ** 2
            loss = torch.sum(loss.reshape(loss.shape[0], -1), dim=-1)
            loss = torch.sqrt(loss)
        elif self.loss_metric == "Huber":
            loss = (D_yt - D_y_ref) ** 2
            loss = torch.sum(loss.reshape(loss.shape[0], -1), dim=-1)
            loss = torch.sqrt(loss + self.c ** 2) - self.c
        elif self.loss_metric == "Cauchy":
            loss = (D_yt - D_y_ref) ** 2
            loss = torch.sum(loss.reshape(loss.shape[0], -1), dim=-1)
            loss = torch.log(1 + loss / (2 * self.c ** 2))
        elif self.loss_metric == "ELatentLPIPS":
            loss = self.elatentlpips(D_yt, D_y_ref)

        loss = loss.reshape(loss.shape[0],)

        return loss


@persistence.persistent_class
class ECDLoss:
    def __init__(self, 
            P_mean=-1.1, P_std=2.0, sigma_data=0.5, 
            q=4, c=0.0, k=8.0, b=1.0, adj='sigmoid', wt='snrpk',
            solver='euler'
            ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
       
        if adj == 'const':
            dist.print0('const adj')
            self.t_to_r = self.t_to_r_const
        elif adj == 'sigmoid':
            dist.print0('sigmoid adj')
            self.t_to_r = self.t_to_r_sigmoid
        else:
            raise ValueError(f'Unknow schedule type {adj}!')
        
        if wt == 'snrpk':
            self.wt_fn = self.snrplusk_wt
        else:
            raise ValueError(f'Unknow wt fn type {adj}!')

        self.q = q
        self.stage = 0
        self.ratio = 0.
        
        self.k = k
        self.b = b

        self.c = c

        self.solver = solver
        dist.print0(f'Wt: {wt}, P_mean: {self.P_mean}, P_std: {self.P_std}, q: {self.q}, k {self.k}, b {self.b}, c: {self.c}, solver: {self.solver}')

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

    def snrplusk_wt(self, t, r):
        # SNR(t) + k = 1/t**2 + k
        wt = (t ** 2 + self.sigma_data ** 2) / (t * self.sigma_data) ** 2
        return wt
    
    def euler_solver(self, x, t, r, net_ref):
        denoiser = net_ref(x, t)
        score = denoiser - x 
        return x - (r - t) / t * score

    def heun_solver(self, x, t, r, net_ref):
        denoiser = net_ref(x, t)
        d = (x - denoiser) / t

        x_prime = x + (r - t) * d
        denoiser_prime = net_ref(x_prime, r)
        d_prime = (x_prime - denoiser_prime) / r

        return x + (r - t) / 2 * (d + d_prime)

    def __call__(self, net, net_ref, images, labels=None):
        # t ~ p(t) and r ~ p(r|t, iters) (Mapping fn)
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        t = (rnd_normal * self.P_std + self.P_mean).exp()
        r = self.t_to_r(t)

        # Augmentation
        x_0 = images
        
        # Shared noise direction
        eps = torch.randn_like(x_0)
        x_t = x_0 + eps * t
        # x_r = x_0 + eps * r
        net_ref_fn = lambda x, t: net_ref(x, t, labels)
        if self.solver == 'euler':
            x_r = self.euler_solver(x_t, t, r, net_ref_fn)
        elif self.solver == 'heun':
            x_r = self.heun_solver(x_t, t, r, net_ref_fn)
        
        # Shared Dropout Mask
        rng_state = torch.cuda.get_rng_state()
        fx_t = net(x_t, t, labels)
        
        if r.max() > 0:
            torch.cuda.set_rng_state(rng_state)
            with torch.no_grad():
                with disable_forced_wn(net):
                    fx_r = net(x_r, r, labels)
            
            mask = r > 0
            fx_r = torch.nan_to_num(fx_r)
            fx_r = mask * fx_r + (~mask) * x_0
        else:
            fx_r = x_0

        # L2 Loss
        loss = (fx_t - fx_r) ** 2
        loss = torch.sum(loss.reshape(loss.shape[0], -1), dim=-1)
        
        if self.c > 0:
            loss = torch.sqrt(loss + self.c ** 2) - self.c
        else:
            loss = torch.sqrt(loss)
        
        wt = self.wt_fn(t, r)
        return loss * wt.flatten()

@persistence.persistent_class
class ECDGuidLoss:
    def __init__(self, 
            P_mean=-1.1, P_std=2.0, sigma_data=0.5, 
            q=4, c=0.0, k=8.0, b=1.0, adj='sigmoid', wt='snrpk',
            solver='euler', guid=1.0,
            loss_metric="Huber",
            ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
       
        if adj == 'const':
            dist.print0('const adj')
            self.t_to_r = self.t_to_r_const
        elif adj == 'sigmoid':
            dist.print0('sigmoid adj')
            self.t_to_r = self.t_to_r_sigmoid
        else:
            raise ValueError(f'Unknow schedule type {adj}!')
        
        if wt == 'snrpk':
            self.wt_fn = self.snrplusk_wt
        else:
            raise ValueError(f'Unknow wt fn type {adj}!')

        self.q = q
        self.stage = 0
        self.ratio = 0.
        
        self.k = k
        self.b = b

        self.c = c

        self.solver = solver
        self.guid = guid
        self.loss_metric = loss_metric

        if self.loss_metric == "ELatentLPIPS":
            from elatentlpips import ELatentLPIPS2
            self.elatentlpips = ELatentLPIPS2(
                pretrained_latent_lpips_path="vgg.pt", 
                aug_type="bgcfnc").to("cuda")

        dist.print0(f'Wt: {wt}, P_mean: {self.P_mean}, P_std: {self.P_std}, q: {self.q}, k {self.k}, b {self.b}, c: {self.c}, solver: {self.solver}, guid: {self.guid}, loss_metric: {self.loss_metric}')

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

    def snrplusk_wt(self, t, r):
        # SNR(t) + k = 1/t**2 + k
        wt = (t ** 2 + self.sigma_data ** 2) / (t * self.sigma_data) ** 2
        return wt
    
    def euler_solver(self, x, t, r, net_ref):
        denoiser = net_ref(x, t)
        score = denoiser - x 
        return x - (r - t) / t * score

    def heun_solver(self, x, t, r, net_ref):
        denoiser = net_ref(x, t)
        d = (x - denoiser) / t

        x_prime = x + (r - t) * d
        denoiser_prime = net_ref(x_prime, r)
        d_prime = (x_prime - denoiser_prime) / r

        return x + (r - t) / 2 * (d + d_prime)

    def __call__(self, net, net_ref, g_net, images, labels=None):
        # t ~ p(t) and r ~ p(r|t, iters) (Mapping fn)
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        t = (rnd_normal * self.P_std + self.P_mean).exp()
        r = self.t_to_r(t)

        # Augmentation
        x_0 = images
        
        # Shared noise direction
        eps = torch.randn_like(x_0)
        x_t = x_0 + eps * t
        # x_r = x_0 + eps * r
        net_ref_fn = lambda x, t: g_net(x, t, labels).lerp(net_ref(x, t, labels), self.guid)
        if self.solver == 'euler':
            x_r = self.euler_solver(x_t, t, r, net_ref_fn)
        elif self.solver == 'heun':
            x_r = self.heun_solver(x_t, t, r, net_ref_fn)
        
        # Shared Dropout Mask
        rng_state = torch.cuda.get_rng_state()
        fx_t = net(x_t, t, labels)
        
        if r.max() > 0:
            torch.cuda.set_rng_state(rng_state)
            with torch.no_grad():
                with disable_forced_wn(net):
                    fx_r = net(x_r, r, labels)
            
            mask = r > 0
            fx_r = torch.nan_to_num(fx_r)
            fx_r = mask * fx_r + (~mask) * x_0
        else:
            fx_r = x_0

        # L2 Loss
        loss = (fx_t - fx_r) ** 2
        loss = torch.sum(loss.reshape(loss.shape[0], -1), dim=-1)
        
        if self.c > 0 and self.loss_metric == "Huber":
            loss = torch.sqrt(loss + self.c ** 2) - self.c
        elif self.c > 0 and self.loss_metric == "Cauchy":
            loss = torch.log(1 + loss / (2 * self.c ** 2))
        elif self.loss_metric == "ELatentLPIPS":
            loss = self.elatentlpips(fx_t, fx_r)
        else:
            loss = torch.sqrt(loss)
        
        wt = self.wt_fn(t, r)
        return loss * wt.flatten()

