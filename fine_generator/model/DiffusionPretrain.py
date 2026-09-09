import torch
import torch.nn.functional as F

from metrics.evaluation_metrics import chamfer_distance_l1, chamfer_distance_l2
from model.Diffusion import VarianceSchedule, TimeEmbedding
from model.Decoder_Component import *
from pointnet2_ops import pointnet2_utils
import numpy as np
from timm.models.layers import DropPath,trunc_normal_
from model.Encoder import *
from model.Encoder_Component import *
import time

"""
Prediction on the masked patches only
w/ diffusion process
Use FC layer as the mask token convertor
"""



class Diff_Point_MAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.trans_dim = config.trans_dim
        self.group_size = 32
        self.num_group = 32
        
        self.num_output = config.diffusion_output_size
        self.num_channel = 3
        self.drop_path_rate = config.drop_path_rate
        self.mask_token = nn.Conv1d((self.num_channel * self.num_output) // 512, self.trans_dim, 1)
        
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.decoder_depth = config.decoder_depth
        self.decoder_num_heads = config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        
        self.MAE_decoder = Transformer(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)
        self.var = VarianceSchedule(
            num_steps=config.num_steps,
            beta_1=config.beta_1,
            beta_T=config.beta_T,
            mode=config.sched_mode
        )

        
        self.increase_dim = nn.Sequential(
            nn.Conv1d(384, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 96, 1)
            )

        self.timestep = config.num_steps
        self.beta_1 = config.beta_1
        self.beta_T = config.beta_T

        self.betas = self.linear_schedule(timesteps=self.timestep)

        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, axis=0)
        self.alpha_bar_t_minus_one = F.pad(self.alpha_bar[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - self.alpha_bar)
        self.sigma = self.betas * (1.0 - self.alpha_bar_t_minus_one) / (1.0 - self.alpha_bar)
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_alpha_bar_minus_one = torch.sqrt(self.alpha_bar_t_minus_one)

        self.time_emb = nn.Sequential(
            TimeEmbedding(self.trans_dim),
            nn.Linear(self.trans_dim, self.trans_dim),
            nn.ReLU()
        )
        
        
        self.group_divider = Group(32, 32)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = chamfer_distance_l1
        elif loss_type == 'cdl2':
            self.loss_func = chamfer_distance_l2
        elif loss_type == 'mse':
            self.loss_func = F.mse_loss
        else:
            raise NotImplementedError
        
    def linear_schedule(self, timesteps):
        return torch.linspace(self.beta_1, self.beta_T, timesteps)

    def get_index_from_list(self, vals, t, x_shape):
        b = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(b, *((1,) * (len(x_shape) - 1))).to(t.device)

    def forward_diffusion(self, x_0, t):
        """
        Adding noise to the original input for
        the forward diffusion process. The noise level
        calculated based on current timestep t.

        :param x_0: The original masked input.
                    [B, NM, C] where
                    B = Batch size,
                    NM = Number of point in masked patches,
                    C = Data Channels (3 for current task)
        :param t: The current timestep
        :return: The noisy masked input at timestep t.
                 [B, NM, C]
        """
        noise = torch.randn_like(x_0).to(x_0.device)
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_bar, t, x_0.shape).to(x_0.device)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_bar, t, x_0.shape).to(
            x_0.device)

        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise

    def compute_bbox_loss(self, points, bboxes):
        """
        Computes the bounding box constraint loss.
        
        Args:
            points (torch.Tensor): Generated point cloud (B, N, 3)
            bboxes (torch.Tensor): Target bounding boxes (B, 6) -> [min_x, y, z, max_x, y, z]
            
        Returns:
            torch.Tensor: Scalar loss value
        """
        B, N, _ = points.shape
        device = points.device

        b_min = bboxes[:, :3].unsqueeze(1) # [B, 1, 3]
        b_max = bboxes[:, 3:].unsqueeze(1) # [B, 1, 3]

        loss_max = torch.relu(points - b_max)

        loss_min = torch.relu(b_min - points)

        total_loss = (loss_max + loss_min).sum(dim=(1, 2))

        return total_loss.mean()        
        
    def forward(self, x_vis, pos_vis, gt, t, pred_coarse, bbox):
        
        B, M, C = x_vis.shape
        N = 384 - M
        
        ts = self.time_emb(t.to(x_vis.device)).unsqueeze(1).expand(-1, 384, -1)
        
        pos_full = torch.cat([pos_vis, pred_coarse], dim=1)
        pos_full = self.decoder_pos_embed(pos_full)
        
        x_t, noise = self.forward_diffusion(gt, t)
        x_t = self.group_divider.get_neighborhood(x_t, pred_coarse).reshape(B, -1, 3)
        mask_token = self.mask_token(x_t.reshape(B, N, -1).transpose(1, 2)).transpose(1, 2).to(x_vis.device)
        x_full = torch.cat([x_vis, mask_token], dim=1)
        

        x_prior_rec = self.MAE_decoder(x_full, pos_full, N, ts)
        x_rec = self.increase_dim(x_prior_rec.transpose(1, 2)).transpose(1, 2).reshape(B, -1, 3)
        
        return self.loss_func(x_rec, gt) + self.compute_bbox_loss(x_rec, bbox)
        
    def sampling_t(self, noisy_t, x_vis, pos_vis, pos_msk, t):
        """
        Reverse sampling at timestep t.
        Input noisy level at timestep t,
        return noisy level at timestep t-1.

        :param noisy_t: The noisy masked patches at timestep t.
               [B, NM, C]
        :param t: Timestep.
        :param mask: The mask indicator. [B, G]
        :param center: The center points. [B, G, C]
        :param x_vis: The latent of visible patches. [B, V, L]
        :return: The noisy masked patches at timestep t-1. [B, NM, C]
        """
        B, M, C = x_vis.shape
        N = 384 - M
        
        ts = self.time_emb(t.to(x_vis.device)).unsqueeze(1).expand(-1, 384, -1)
        betas_t = self.get_index_from_list(self.betas, t, noisy_t.shape).to(x_vis.device)
        
        pos_full = torch.cat([pos_vis, pos_msk], dim=1)
        pos_full = self.decoder_pos_embed(pos_full)
        
        noisy_t = self.group_divider.get_neighborhood(noisy_t, pos_msk).reshape(B, -1, 3)
        mask_token = self.mask_token(noisy_t.reshape(B, N, -1).transpose(1, 2)).transpose(1, 2).to(x_vis.device)
        x_full = torch.cat([x_vis, mask_token], dim=1)
        
        x_prior_rec = self.MAE_decoder(x_full, pos_full, N, ts)
        x_rec = self.increase_dim(x_prior_rec.transpose(1, 2)).transpose(1, 2).reshape(B, -1, 3)
        
        alpha_bar_t = self.get_index_from_list(self.alpha_bar, t, noisy_t.shape).to(x_vis.device)
        alpha_bar_t_minus_one = self.get_index_from_list(self.alpha_bar_t_minus_one, t, noisy_t.shape).to(x_vis.device)
        sqrt_alpha_t = self.get_index_from_list(self.sqrt_alphas, t, noisy_t.shape).to(x_vis.device)
        sqrt_alphas_bar_t_minus_one = self.get_index_from_list(self.sqrt_alpha_bar_minus_one, t, noisy_t.shape).to(
            x_vis.device)

        model_mean = (sqrt_alpha_t * (1 - alpha_bar_t_minus_one)) / (1 - alpha_bar_t) * noisy_t + (
                    sqrt_alphas_bar_t_minus_one * betas_t) / (1 - alpha_bar_t) * x_rec

        sigma_t = self.get_index_from_list(self.sigma, t, noisy_t.shape).to(x_vis.device)

        if t == 0:
            return model_mean
        else:
            return model_mean + torch.sqrt(sigma_t) * x_rec

    def sampling(self, x_vis, pos_vis, pred_coarse, noise_patch=None):
        """
        Sampling the masked patches from Gaussian noise.

        :param x_vis: The latent of visible patches.
               [B, V, L]
               B = Batch size,
               V = Visible patches size,
               L = Latent size.
        :param mask: The mask indicator.
               [B, G]
               B = Batch size,
               G = Group (Visible patches + Masked patches) size.
        :param center: The center points.
               [B, G, C]
               B = Batch size,
               G = Group size,
               C = Data Channels (3 for current task)
        :param trace: Boolean, False by default.
               if true: return all reverse diffusion steps.
               else: return the last step only.
        :param noise_patch: The pre-defined noises, None by default.
        :return: See param trace.
        """
        
        B, M, C = x_vis.shape
        if noise_patch is None:
            noise_patch = torch.randn((B, (384 - M) * self.group_size, 3)).to(x_vis.device)

        for i in range(0, self.timestep)[::-1]:
            t = torch.full((1,), i, device=x_vis.device)
            noise_patch = self.sampling_t(noise_patch, x_vis, pos_vis, pred_coarse, t)

        return noise_patch.reshape(B, -1, 3)
