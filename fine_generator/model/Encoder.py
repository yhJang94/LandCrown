import torch.nn.functional as F

from metrics.evaluation_metrics import chamfer_distance_l1, chamfer_distance_l2
from model.Encoder_Component import *
from utils import misc
import pdb



class Crown_Encoder_Module(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.trans_dim = config.trans_dim
        self.AE_encoder = PointTransformer(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.num_output = 12288
        self.num_channel = 3
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.group_divider = Group(num_group=32, group_size=32)
        # self.increase_dim = nn.Sequential(
        #     nn.Conv1d(self.trans_dim, (self.num_channel * self.num_output) // self.num_group, 1)
        # )
        
        self.query_num = 384
        self.num_query = 512

        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 96, 1)
        )
        self.loss = config.loss
        self.build_loss_func(self.loss)

        trunc_normal_(self.mask_token, std=.02)
        
        self.num_labels = 28
        self.register_buffer('base_indices', torch.arange(self.num_labels))
        self.class_embedding = nn.Embedding(self.num_labels, config.trans_dim)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = chamfer_distance_l1
        elif loss_type == 'cdl2':
            self.loss_func = chamfer_distance_l2
        elif loss_type == 'mse':
            self.loss_func = F.mse_loss
        else:
            raise NotImplementedError

    def forward(self, input, cond_idx):
        input = input.float()
        B, _, _ = input.shape
        input_reshaped = input.view(B*(self.num_labels-1), 1024, 3)
        indices_reshaped = cond_idx.reshape(-1)
        neighborhood, center = self.group_divider(input_reshaped)
        class_emb = self.class_embedding(indices_reshaped).unsqueeze(1).expand(-1, self.group_size, -1)
        x_vis = self.AE_encoder(neighborhood, center, class_emb)
        pred = self.increase_dim(x_vis.transpose(1, 2)).transpose(1, 2).reshape(B*(self.num_labels-1), -1, 3)
        loss = self.loss_func(pred, input.reshape(B*(self.num_labels-1), -1, 3))
        return x_vis.reshape(B, -1, self.trans_dim), center.reshape(B, -1, 3), pred, loss
    
    def encode(self, input, cond_idx):
        input = input.float()
        B, _, _ = input.shape
        input_reshaped = input.view(B*(self.num_labels-1), 1024, 3)
        indices_reshaped = cond_idx.reshape(-1)
        neighborhood, center = self.group_divider(input_reshaped)
        class_emb = self.class_embedding(indices_reshaped).unsqueeze(1).expand(-1, self.group_size, -1)
        x_vis = self.AE_encoder(neighborhood, center, class_emb)
        pred = self.increase_dim(x_vis.transpose(1, 2)).transpose(1, 2).reshape(B*(self.num_labels-1), -1, 3)
        loss = self.loss_func(pred, input.reshape(B*(self.num_labels-1), -1, 3))
        return x_vis.reshape(B, -1, self.trans_dim), center.reshape(B, -1, 3), pred, loss