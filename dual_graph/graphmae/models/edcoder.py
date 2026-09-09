import os
from typing import Optional
from itertools import chain
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import dgl

from .gin import GIN
from .gat import GAT
from .gcn import GCN
from .dot_gat import DotGAT
from .loss_func import sce_loss
from graphmae.utils import create_norm, drop_edge
import torch.nn.functional as F
# from pointnet2_ops import pointnet2_utils
from geomloss import SamplesLoss

UPPER_ARCH_ORDER = [5, 4, 3, 2, 1, 0, 7, 8, 9, 10, 11, 12]
LOWER_ARCH_ORDER = [26, 25, 24, 23, 22, 21, 14, 15, 16, 17, 18, 19]

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARCH_TEMPLATE_PATH = os.environ.get(
    "ARCH_TEMPLATE_PATH",
    os.path.join(_REPO_ROOT, "debug", "arch_templates", "arch_templates_global.npz"),
)
ARCH_TEMPLATE_LABELS = ["Square", "Ovoid", "Tapered", "Omega"]


def distChamfer(a, b):
        
    x, y = a, b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    
    diag_ind = torch.arange(0, num_points).to(a).long()
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    P = torch.where(P < 0, torch.abs(P), P)
    return P.min(1)[0], P.min(2)[0]

def chamfer_distance_l2(a, b):
    d1, d2 = distChamfer(a, b)
    return torch.mean(d1) + torch.mean(d2)

def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead, nhead_out, attn_drop, negative_slope=0.2, concat_out=True) -> nn.Module:
    if m_type == "gat":
        mod = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=True,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "dotgat":
        mod = DotGAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            residual=True,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gin":
        mod = GIN(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=norm,
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gcn":
        mod = GCN(
            in_dim=in_dim, 
            num_hidden=num_hidden, 
            out_dim=out_dim, 
            num_layers=num_layers, 
            dropout=dropout, 
            activation=activation, 
            residual=residual, 
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding")
        )
    elif m_type == "mlp":
        # * just for decoder 
        mod = nn.Sequential(
            nn.Linear(in_dim, num_hidden),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_hidden, out_dim)
        )
    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError
    
    return mod


class PreModel(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,
            nhead_out: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            residual: bool,
            norm: Optional[str],
            mask_rate: float = 0.3,
            encoder_type: str = "gat",
            decoder_type: str = "gat",
            loss_fn: str = "mse",
            drop_edge_rate: float = 0.0,
            replace_rate: float = 0.1,
            alpha_l: float = 2,
            concat_hidden: bool = False,
         ):
        super(PreModel, self).__init__()
        self._output_hidden_size = num_hidden

        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        if encoder_type in ("gat", "dotgat"):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1

        dec_in_dim = num_hidden

        # build encoder
        self.encoder = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden // 2,
            num_layers=num_layers-1,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            # dropout=feat_drop,
            dropout=0.0,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=False,
            norm=norm,
        )
        
        self.encoder2 = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim // 2,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden // 4,
            num_layers=num_layers-1,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            # dropout=feat_drop,
            dropout=0.0,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=False,
            norm=norm,
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        
        self.decoder_down = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        
        self.decoder2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )
        
        self.decoder_down2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )

        self.encoder_to_decoder = nn.Sequential(
                nn.Linear(dec_in_dim // 4, dec_in_dim // 4, bias=False),
                nn.LayerNorm(dec_in_dim // 4), # 여기서 한 번 잡아줌
                nn.ReLU()                      # 필요하다면 활성화 함수 추가
            )

        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)
        self.cd_criterion = chamfer_distance_l2
        
        self.coord_encoder = nn.Linear(3, num_hidden)
        self.class_embedding = nn.Embedding(6, num_hidden)
        self.input_norm = nn.LayerNorm(num_hidden)

        
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        
        self.teeth_num = 28
        self.learnable_offset = nn.Parameter(torch.randn(self.teeth_num, 6, 3) * 0.001)

        self.register_buffer('upper_arch_order', torch.tensor(UPPER_ARCH_ORDER, dtype=torch.long))
        self.register_buffer('lower_arch_order', torch.tensor(LOWER_ARCH_ORDER, dtype=torch.long))

        arch_template_data = np.load(ARCH_TEMPLATE_PATH)
        arch_templates = np.stack([
            np.stack([arch_template_data[f'{arch}_{label}'] for label in ARCH_TEMPLATE_LABELS], axis=0)
            for arch in ['upper', 'lower']
        ], axis=0)  # (2 arch, 4 type, 12, 3)
        self.register_buffer('arch_templates', torch.tensor(arch_templates, dtype=torch.float32))

        self.disp_scale = nn.Parameter(torch.tensor(0.1))

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
            return criterion
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def umeyama_align_batched(self, P, Q, mask):

        with torch.no_grad():
            maskf = mask.float().unsqueeze(-1)  # (...,N,1)
            n_valid = maskf.sum(dim=-2, keepdim=True).clamp(min=1e-6)  # (...,1,1)
            P_mean = (P * maskf).sum(dim=-2, keepdim=True) / n_valid  # (...,1,3)
            Q_mean = (Q * maskf).sum(dim=-2, keepdim=True) / n_valid  # (...,1,3)
            Pc = (P - P_mean) * maskf
            Qc = (Q - Q_mean) * maskf

            H = torch.matmul(Pc.transpose(-1, -2), Qc) / n_valid  # (...,3,3)
            U, S, Vt = torch.linalg.svd(H)
            det_sign = torch.det(torch.matmul(Vt.transpose(-1, -2), U.transpose(-1, -2)))
            d = torch.sign(det_sign)
            ones = torch.ones_like(d)
            diag_vals = torch.stack([ones, ones, d], dim=-1)  # (...,3)
            diag = torch.diag_embed(diag_vals)  # (...,3,3)
            R = torch.matmul(torch.matmul(Vt.transpose(-1, -2), diag), U.transpose(-1, -2))  # (...,3,3)

            var_P = (Pc.pow(2).sum(dim=(-1, -2)) / n_valid.squeeze(-1).squeeze(-1)).clamp(min=1e-6)  # (...)
            scale_num = (S * diag_vals).sum(dim=-1)  # (...)
            s = scale_num / var_P  # (...)

            Rt_Pmean = torch.matmul(R, P_mean.squeeze(-2).unsqueeze(-1)).squeeze(-1)  # (...,3)
            t = Q_mean.squeeze(-2) - s.unsqueeze(-1) * Rt_Pmean  # (...,3)
        return R, s, t

    def predict_seed_arch_template(self, x_masked, idx):
        B = x_masked.shape[0]
        device = x_masked.device
        num_teeth = self.teeth_num
        points_per_tooth = x_masked.shape[1] // num_teeth

        x_reshaped = x_masked.reshape(B, num_teeth, points_per_tooth, 3)
        case_centroids = x_reshaped.mean(dim=2)  # (B, 28, 3)
        valid = (x_reshaped.std(dim=2).sum(dim=-1) > 1e-6)  # (B, 28)

        is_upper = idx < 14  # (B,)
        order = torch.where(
            is_upper.unsqueeze(-1),
            self.upper_arch_order.unsqueeze(0).expand(B, -1),
            self.lower_arch_order.unsqueeze(0).expand(B, -1),
        )  # (B, 12)

        patient_pts = torch.gather(case_centroids, 1, order.unsqueeze(-1).expand(-1, -1, 3))  # (B,12,3)
        patient_valid = torch.gather(valid, 1, order)  # (B,12)
        target_pos = (order == idx.unsqueeze(-1)).long().argmax(dim=-1)  # (B,) 

        arch_sel = (~is_upper).long()  # 0=upper, 1=lower
        templates = self.arch_templates.to(device)[arch_sel]  # (B, 4, 12, 3)

        P = templates
        Q = patient_pts.unsqueeze(1).expand(-1, 4, -1, -1)  # (B,4,12,3)
        mask = patient_valid.unsqueeze(1).expand(-1, 4, -1)  # (B,4,12)

        R, s, t = self.umeyama_align_batched(P, Q, mask)  # (B,4,3,3),(B,4),(B,4,3)
        transformed = s.unsqueeze(-1).unsqueeze(-1) * torch.matmul(P, R.transpose(-1, -2)) + t.unsqueeze(-2)  # (B,4,12,3)

        maskf = mask.float()
        resid_sq = (((transformed - Q) ** 2).sum(dim=-1) * maskf).sum(dim=-1)  # (B,4)
        n_valid = maskf.sum(dim=-1).clamp(min=1.0)  # (B,4)
        resid = torch.sqrt(resid_sq / n_valid)  # (B,4)

        worst_val, worst_idx = resid.max(dim=-1, keepdim=True)  # (B,1)
        keep_mask = torch.ones_like(resid, dtype=torch.bool)
        keep_mask.scatter_(1, worst_idx, False)
        lin = torch.clamp(worst_val - resid, min=1e-8) * keep_mask.float()  # (B,4)
        weights = lin / lin.sum(dim=-1, keepdim=True)

        target_transformed = torch.gather(
            transformed, 2, target_pos.view(B, 1, 1, 1).expand(-1, 4, 1, 3)
        ).squeeze(2)  # (B,4,3)
        pred_centroid = (weights.unsqueeze(-1) * target_transformed).sum(dim=1)  # (B,3)

        return pred_centroid, case_centroids

    def forward(self, g, x, cr, idx, coarse_gt, is_train=False):
        # ---- attribute reconstruction ----
        x_rec, x_init, loss, mse, seed_loss, seed_mse_loss, hidden, use_x, pred_centroid = self.mask_attr_prediction(g, x, cr, idx, coarse_gt, is_train)
        loss_item = {"loss": loss.item(), "mse": mse.item(), "seed_loss": seed_loss.item(), "seed_mse_loss": seed_mse_loss.item()}
        return x_rec, x_init, loss, loss_item, hidden, use_x, pred_centroid

    def mask_attr_prediction(self, g, x, g2, idx, coarse_gt, is_train):

        mask_nodes = (g.ndata["mask"] == 1).nonzero(as_tuple=True)[0]
        use_g = g
        use_x = x.clone()

        B = len(g.batch_num_nodes().tolist())
        g2_feat = g2.ndata["feat"].clone()

        pred_centroid, case_centroids = self.predict_seed_arch_template(g2_feat.reshape(B, -1, 3), idx)
        real_centroid = coarse_gt.mean(dim=1)  # (B, 3)
        # --- seed_loss / seed_mse_loss disabled ---
        # seed_loss = F.mse_loss(pred_centroid, real_centroid)
        seed_loss = torch.zeros((), device=x.device)

        center = real_centroid if is_train else pred_centroid

        labels = g.ndata["label"][mask_nodes]
        tooth_nums = g.ndata["tooth_num"][mask_nodes]

        node_batch_id = dgl.broadcast_nodes(g, torch.arange(B, device=g.device))
        sample_idx = node_batch_id[mask_nodes]

        picked_offsets = self.learnable_offset[tooth_nums, labels]
        picked_centers = center[sample_idx]

        all_tooth_nums = g.ndata["tooth_num"]
        all_labels = g.ndata["label"]
        # all_offsets = self.learnable_offset[all_tooth_nums, all_labels]

        target_flag = (g.ndata["mask"] == 1).squeeze(-1)
        # node_centroid = case_centroids[node_batch_id, all_tooth_nums].clone()
        # node_centroid[target_flag] = pred_centroid[node_batch_id[target_flag]]

        # seed_all = node_centroid + all_offsets
        # seed_mse_loss = F.mse_loss(seed_all, x)
        seed_mse_loss = torch.zeros((), device=x.device)


        use_x[mask_nodes] = 0.0
        use_x[mask_nodes] += picked_centers + picked_offsets
        
        node_center_for_input = case_centroids[node_batch_id, all_tooth_nums].clone()
        node_center_for_input[target_flag] = center[node_batch_id[target_flag]]

        # landmark class embedding
        label = self.class_embedding(g.ndata["label"].long())
        embed_x = self.coord_encoder(use_x - node_center_for_input)
        input_x = self.input_norm(embed_x + label)
        
        # encode in GAT Layer
        enc_rep = self.encoder(use_g, input_x)
        enc_rep2 = self.encoder2(use_g, enc_rep)
        
        # bridge
        rep = self.encoder_to_decoder(enc_rep2)
        
        # upsampling
        dec_rep = self.decoder(rep)  
        # skip connection
        recon_cat = torch.cat([enc_rep, dec_rep], dim=-1) 
        # feature fusion
        recon_cat = self.decoder_down(recon_cat)
        
        # upsampling
        dec_rep2 = self.decoder2(recon_cat)
        # skip connection
        recon_cat2 = torch.cat([input_x, dec_rep2], dim=-1)
        # feature fusion
        recon_cat2 = self.decoder_down2(recon_cat2)
        
        # projection
        disp = self.head(recon_cat2)
        recon = self.disp_scale * disp + use_x

        # bring missing part
        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]

        mse = self.criterion(x_rec, x_init)

        # loss = mse + seed_loss + seed_mse_loss
        loss = mse

        lm_init = use_x.clone()
        lm_init[mask_nodes] = x_rec

        # keep for usage in fine generator
        return x_rec, x_init, loss, mse, seed_loss, seed_mse_loss, [rep, recon_cat, recon_cat2], lm_init, pred_centroid

    def embed(self, g, x):
        rep = self.encoder(g, x)
        return rep


    @property
    def enc_params(self):
        return self.encoder.parameters()
    
    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])

class PreModel2(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,
            nhead_out: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            residual: bool,
            norm: Optional[str],
            mask_rate: float = 0.3,
            encoder_type: str = "gat",
            decoder_type: str = "gat",
            loss_fn: str = "mse",
            drop_edge_rate: float = 0.0,
            replace_rate: float = 0.1,
            alpha_l: float = 2,
            concat_hidden: bool = False,
         ):
        super(PreModel2, self).__init__()

        self._output_hidden_size = num_hidden

        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        if encoder_type in ("gat", "dotgat"):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1


        dec_in_dim = num_hidden
        dec_num_hidden = num_hidden // nhead_out if decoder_type in ("gat", "dotgat") else num_hidden 

        # build encoder
        self.encoder = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=num_hidden,
            num_hidden=num_hidden,
            out_dim=num_hidden // 2,
            num_layers=num_layers-1,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=False,
            activation=activation,
            # dropout=feat_drop,
            dropout=0.0,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=False,
            norm=norm,
        )
        
        self.encoder2 = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=num_hidden // 2,
            num_hidden=num_hidden,
            out_dim=num_hidden // 4,
            num_layers=num_layers-1,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=False,
            activation=activation,
            # dropout=feat_drop,
            dropout=0.0,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=False,
            norm=norm,
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        
        self.decoder_down = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        
        self.decoder2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )
        
        self.decoder2_down = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )
        
        base_indices = torch.arange(28)
        fixed_indices = torch.repeat_interleave(base_indices, 128)
        self.register_buffer('indices', fixed_indices)
        self.encoder_to_decoder = nn.Linear(dec_in_dim // 4, dec_in_dim // 4, bias=False)

        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)
        self.cd_criterion = chamfer_distance_l2
        # self.cd_criterion = SamplesLoss(loss="sinkhorn", p=2, blur=.05, scaling=0.9)
        
        self.coord_encoder = nn.Linear(3, num_hidden)
        self.tooth_embedding = nn.Embedding(28, num_hidden)
        self.input_norm = nn.LayerNorm(num_hidden)
        
        self.head = nn.Sequential(
            nn.Linear(256, 128), # 1024로 키워서 공간 정보를 꽉 잡음
            nn.ReLU(),
            nn.Linear(128, 3),
        )
        
        self.pos_128 = self._make_pe(128)
        self.pos_256 = self._make_pe(256)
        
        self.stage2 = ToothDecoderBlock(128)
        self.stage3 = ToothDecoderBlock(256)
        
        self.teeth_num = 28
        self.learnable_offset = nn.Parameter(torch.randn(self.teeth_num, 128, 3) * 0.001)
        self.disp_scale = nn.Parameter(torch.tensor(0.1))

        self.rep_weight_init = 0.3
        self.rep_weight_min = 0.05
        self.rep_anneal_epochs = 100
        self.rep_k = 6

        self.extent_weight = 0.3

    def current_rep_weight(self, epoch):
        frac = min(1.0, epoch / self.rep_anneal_epochs)
        return self.rep_weight_init + (self.rep_weight_min - self.rep_weight_init) * frac

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion2 = nn.MSELoss()
            criterion = nn.L1Loss()
            return criterion, criterion2
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion
    
    def masking(self, use_x, batch2, idx, interpol):
        
        current_offset = 0
    
        for i, num_nodes in enumerate(batch2):            
            start = current_offset + (idx[i] * 128)
            end = current_offset + ((idx[i] + 1) * 128)
            use_x[start:end] = interpol[i]
            current_offset += num_nodes
        
        return use_x
    
    def bring_missing_part(self, gt, idx):
        
        B, N, C = gt.shape
        num_teeth = 28
        points_per_tooth = 128

        gt_segmented = gt.reshape(B, num_teeth, points_per_tooth, C)

        batch_indices = torch.arange(B, device=gt.device)
        
        missing_part = gt_segmented[batch_indices, idx]

        return missing_part

    def repulsion_loss(self, points, ref_points, k=6):

        N = points.shape[1]
        eye_mask = torch.eye(N, device=points.device, dtype=torch.bool).unsqueeze(0)

        with torch.no_grad():
            ref_dist = torch.cdist(ref_points, ref_points)
            ref_dist = ref_dist.masked_fill(eye_mask, float('inf'))
            h = ref_dist.min(dim=-1)[0].mean(dim=-1)  # (B,) 샘플별 seed 평균 최근접 거리
            h = h.clamp(min=1e-2)

        jitter = torch.randn_like(points) * 1e-5
        pts_j = points + jitter
        sq_dist = (pts_j.unsqueeze(2) - pts_j.unsqueeze(1)).pow(2).sum(-1)  # (B,N,N)
        dist = torch.sqrt(sq_dist + 1e-12)
        dist = dist.masked_fill(eye_mask, float('inf'))
        knn_dist, _ = torch.topk(dist, k=k, dim=-1, largest=False)  # (B,N,k)
        penalty = torch.clamp(1 - knn_dist / h.view(-1, 1, 1), min=0) ** 2
        return penalty.mean()

    def extent_loss(self, points, ref_points):
        """
        points 전체의 스케일(std, 축별)이 seed(ref_points)와 비슷하게 유지되도록 직접 잡아주는 항.
        repulsion loss(local pairwise 간격)와 달리 이건 global scale을 명시적으로 맞춘다.
        seed 쪽으로 gradient가 흘러 들어가면 안 되므로 ref_points는 detach.

        repulsion_loss와 동일한 이유로(점들이 bit-identical하게 collapse되면 var의 gradient가
        정확히 0) distance 계산에만 아주 작은 jitter를 더해 collapse 상태에서도 gradient가 흐르게 함.
        """
        # repulsion_loss와 동일한 이유로, seed std 대비 상대오차로 정규화해서 최댓값을 1.0으로 고정
        # (절대 std^2 차이를 그대로 쓰면 실제 데이터 스케일상 상한이 너무 작아 weight가 무력해짐).
        # 학습 초반엔 seed(learnable_offset)가 randn*0.001로 초기화돼 있어 ref_std가 실제 치아
        # 스케일(~0.02~0.08)보다 훨씬 작을 수 있음 -> 분모가 너무 작아지는 걸 막는 floor 필요
        # (실제로 이 floor 없이 돌렸더니 학습 초반 ext_loss가 1000 이상으로 폭주했음).
        jitter = torch.randn_like(points) * 1e-5
        std = (points + jitter).std(dim=1)
        ref_std = ref_points.detach().std(dim=1).clamp(min=1e-2)
        ratio_sq = ((std - ref_std) / ref_std) ** 2
        return ratio_sq.clamp(max=25.0).mean()

    def forward(self, g, x, idx, hidden, batch2, g1, lm_init, coarse_gt, center, is_train=False, epoch=0):
        # ---- attribute reconstruction ----
        x_rec, x_init, loss, cd_loss, seed_cd_loss, rep_loss, ext_loss = self.mask_attr_prediction(g, x, idx, hidden, batch2, g1, lm_init, coarse_gt, center, is_train, epoch)
        loss_item = {"loss": loss.item(), "cd_loss": cd_loss.item(), "seed_cd_loss": seed_cd_loss.item(), "rep_loss": rep_loss.item(), "ext_loss": ext_loss.item()}
        return x_rec, x_init, loss, loss_item
    
    def reshape_to_batches(self, x, batch2, max_nodes=90):
        """
        x: [Total_Nodes, dim] (예: 10037, 128)
        batch2: 조각별 노드 수 리스트 (예: [80, 79, 82, ...])
        max_nodes: 고정할 노드 수 (90)
        """
        dim = x.shape[-1]
        batch_size = len(batch2)
        
        x_split = torch.split(x, batch2)
        
        x_padded = torch.zeros((batch_size, max_nodes, dim), device=x.device)
        
        mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool, device=x.device)
        
        for i, chunk in enumerate(x_split):
            num_nodes = chunk.size(0)
            actual_fill = min(num_nodes, max_nodes)
            x_padded[i, :actual_fill, :] = chunk[:actual_fill, :]
            mask[i, :actual_fill] = True
            
        return x_padded, mask

    def mask_attr_prediction(self, g, x, idx, kv_feat, batch2, g1, lm_init, coarse_gt, center, is_train=False, epoch=0):

        assert len(batch2) == g.batch_size

        B = len(batch2)
        use_g = g
        use_x = x.clone()

        # center(결손 치아 centroid)는 PreModel에서 이미 계산돼서 넘어옴
        # (학습 시엔 실제 GT centroid로 teacher forcing, 테스트 시엔 PreModel의 seed 예측값)
        interpol = center.unsqueeze(1) + self.learnable_offset[idx]
        # interpol = self.learnable_offset[idx]
        
        x_masking = self.masking(use_x, batch2, idx, interpol)

        # GAT 입력(및 이후 q_pos/kv_pos)용 "치아별 center 상대좌표": 존재하는 치아는 실측 center,
        # target은 이미 넘어온 center(real/pred)를 사용 -> 절대 위치 대신 "자기 치아 안에서의 역할"로 표현
        case_centroids_local = x.reshape(B, 28, 128, 3).mean(dim=2)  # (B,28,3)
        per_tooth_center = case_centroids_local.clone()
        per_tooth_center[torch.arange(B, device=x.device), idx] = center
        point_center_flat = per_tooth_center.unsqueeze(2).expand(-1, -1, 128, -1).reshape(-1, 3)  # (B*3584,3)

        x_encode = self.coord_encoder(x_masking - point_center_flat)

        label_expand = self.indices.unsqueeze(0).expand(B, -1).reshape(B, -1)
        label = self.tooth_embedding(label_expand).reshape(-1, 256)

        input_x = self.input_norm(x_encode + label)

        enc_rep = self.encoder(use_g, input_x)
        enc_rep2 = self.encoder2(use_g, enc_rep)
        rep = self.encoder_to_decoder(enc_rep2)

        # batch reshape for attention calculation
        # kv_mask : for excluding zero padding in later attention. 
        kv_feat[1], kv_mask = self.reshape_to_batches(kv_feat[1], batch2)
        kv_feat[2], _ = self.reshape_to_batches(kv_feat[2], batch2)
        kv_xyz, _ = self.reshape_to_batches(lm_init, batch2)
        
        assert torch.all(kv_feat[1][~kv_mask] == 0)
        assert torch.all(kv_feat[2][~kv_mask] == 0)
        
        batch = g.batch_num_nodes().tolist()
        q_feat = self.reshape_to_batches(rep, batch, batch[0])[0]
        q_xyz = self.reshape_to_batches(x_masking, batch, batch[0])[0]
        skip_enc_rep =  self.reshape_to_batches(enc_rep, batch, batch[0])[0]
        skip_input = self.reshape_to_batches(input_x, batch, batch[0])[0]

        q_center = per_tooth_center.unsqueeze(2).expand(-1, -1, 128, -1).reshape(B, -1, 3)  # (B,3584,3)

        node_batch_id_lm = dgl.broadcast_nodes(g1, torch.arange(B, device=g1.device))
        landmark_tooth_nums = g1.ndata["tooth_num"]
        landmark_centers_flat = per_tooth_center[node_batch_id_lm, landmark_tooth_nums]  # (total_lm_nodes,3)
        kv_center, _ = self.reshape_to_batches(landmark_centers_flat, batch2)  # (B,N_max,3)

        q_xyz_rel = q_xyz - q_center
        kv_xyz_rel = kv_xyz - kv_center

        # upsample
        q_feat = self.decoder(q_feat)
        q_pos_128 = self.pos_128(q_xyz_rel)
        kv_pos = self.pos_128(kv_xyz_rel)
        # attention with landmark features
        q2 = self.stage2(q_feat, kv_feat[1], q_pos_128, kv_pos, ~kv_mask, q_xyz=q_xyz, kv_xyz=kv_xyz)
        # skip connection & feature fusion
        q2 = self.decoder_down(torch.cat([q2, skip_enc_rep], dim=-1))

        # upsample
        q_feat2 = self.decoder2(q2)
        q_pos_256 = self.pos_256(q_xyz_rel)
        kv_pos = self.pos_256(kv_xyz_rel)
        # attention with landmark features
        q3 = self.stage3(q_feat2, kv_feat[2], q_pos_256, kv_pos, ~kv_mask, q_xyz=q_xyz, kv_xyz=kv_xyz)
        # skip connection & feature fusion
        q3 = self.decoder2_down(torch.cat([q3, skip_input], dim=-1))

        # coordinates projection
        disp = self.head(q3)
        recon = self.disp_scale * disp + x_masking.reshape(B, -1, 3)

        # bring missing part
        gt = coarse_gt  
        recon = self.bring_missing_part(recon, idx)

        # optimization
        cd_loss = self.cd_criterion(recon, gt).mean()
        # --- seed_cd / repulsion / extent loss disabled ---
        # seed_cd_loss = self.cd_criterion(interpol, gt).mean()
        # rep_loss = self.repulsion_loss(recon, interpol, k=self.rep_k)
        # rep_w = self.current_rep_weight(epoch)
        # ext_loss = self.extent_loss(recon, interpol)
        seed_cd_loss = torch.zeros((), device=recon.device)
        rep_loss = torch.zeros((), device=recon.device)
        ext_loss = torch.zeros((), device=recon.device)
        # loss = cd_loss + seed_cd_loss + rep_w * rep_loss + self.extent_weight * ext_loss
        loss = cd_loss

        return recon, gt, loss, cd_loss, seed_cd_loss, rep_loss, ext_loss

    def _make_pe(self, out_dim):
        return nn.Sequential(nn.Linear(3, out_dim),
                             # nn.LayerNorm(out_dim),
                             nn.GELU(),
                             nn.Linear(out_dim, out_dim)
                             # nn.LayerNorm(out_dim)
                             )


    @property
    def enc_params(self):
        return self.encoder.parameters()
    
    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])


import torch
import torch.nn as nn
import torch.nn.functional as F

class DistanceBiasCrossAttention(nn.Module):
    """Query-Key 쌍의 실제 3D 거리를 attention score에 직접 bias로 더하는 cross-attention.
    '가까울수록 중요하다'는 방향은 고정(음수 계수)하고, 감쇠 폭(gamma)만 head별로 학습한다.
    절대좌표 PE(덧셈)만으로는 attention이 기하학적 근접성을 스스로 발견해야 하는 한계가 있어서,
    이 관계를 구조적으로 명시해주기 위해 추가함."""
    def __init__(self, dim, nhead):
        super().__init__()
        assert dim % nhead == 0
        self.nhead = nhead
        self.head_dim = dim // nhead
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.gamma_raw = nn.Parameter(torch.zeros(nhead))  # softplus로 항상 양수 유지

    def forward(self, q, kv, q_xyz, kv_xyz, key_padding_mask=None):
        """
        q, kv: (B, Nq/Nk, dim) 콘텐츠 피처
        q_xyz, kv_xyz: (B, Nq/Nk, 3) 실제 3D 좌표 (거리 계산 전용, 절대좌표 PE와 별개)
        key_padding_mask: (B, Nk) True인 곳이 패딩
        """
        B, Nq, _ = q.shape
        Nk = kv.shape[1]
        Q = self.q_proj(q).view(B, Nq, self.nhead, self.head_dim).transpose(1, 2)
        K = self.k_proj(kv).view(B, Nk, self.nhead, self.head_dim).transpose(1, 2)
        V = self.v_proj(kv).view(B, Nk, self.nhead, self.head_dim).transpose(1, 2)

        logits = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim ** 0.5)  # (B,H,Nq,Nk)

        dist_sq = ((q_xyz.unsqueeze(2) - kv_xyz.unsqueeze(1)) ** 2).sum(-1)  # (B,Nq,Nk)
        gamma = F.softplus(self.gamma_raw).view(1, self.nhead, 1, 1)
        logits = logits - gamma * dist_sq.unsqueeze(1)

        if key_padding_mask is not None:
            logits = logits.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(1), float('-inf'))

        attn = torch.softmax(logits, dim=-1)
        out = torch.matmul(attn, V)  # (B,H,Nq,hd)
        out = out.transpose(1, 2).reshape(B, Nq, -1)
        return self.out_proj(out), attn


class ToothDecoderBlock(nn.Module):
    def __init__(self, dim, nhead=8):
        super().__init__()
        # 1. Self-Attention: 정적 노드(384개) 간의 관계 파악
        self.self_attn = nn.MultiheadAttention(dim, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)

        # 2. Cross-Attention: 가변 랜드마크 정보 주입 (거리 기반 bias 포함)
        self.cross_attn = DistanceBiasCrossAttention(dim, nhead)
        self.norm2 = nn.LayerNorm(dim)

        # 3. Feed-Forward (질문하신 Decoder 레이어)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim)
        )

    def forward(self, q, kv, q_pos=None, kv_pos=None, kv_mask=None, q_xyz=None, kv_xyz=None):
        """
        q: 정적 노드 피처 (B, 1536, dim)
        kv: 가변 랜드마크 피처 (B, N_max, dim)
        q_pos: 정적 노드 PE (B, 1536, dim)
        kv_pos: 가변 랜드마크 PE (B, N_max, dim)
        kv_mask: 패딩 마스크 (B, N_max) - True인 곳이 패딩
        q_xyz, kv_xyz: 거리 bias 계산용 실제 3D 좌표
        """

        # --- Step 1: Self-Attention (치아 간 배열 정리) ---
        q_with_pos = q + q_pos if q_pos is not None else q
        attn1, _ = self.self_attn(q_with_pos, q_with_pos, q)
        q = self.norm1(q + attn1)

        # --- Step 2: Cross-Attention (랜드마크 정보 흡수, 거리 bias 포함) ---
        k = kv + kv_pos if kv_pos is not None else kv
        attn2, _ = self.cross_attn(q + q_pos, k, q_xyz, kv_xyz, key_padding_mask=kv_mask)
        q = self.norm2(q + attn2)

        # --- Step 3: Decoder FFN (차원 확장 및 형태 확정) ---
        q = q + self.ffn(q)

        return q