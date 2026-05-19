import os
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import cv2

from ProtoSeg.lib_PMSAD.models.backbones.backbone_selector import BackboneSelector
from ProtoSeg.lib_PMSAD.models.tools.module_helper import ModuleHelper
from ProtoSeg.lib_PMSAD.models.modules.projection import ProjectionHead
from ProtoSeg.lib_PMSAD.utils.tools.logger import Logger as Log
from ProtoSeg.lib_PMSAD.models.modules.hanet_attention import HANet_Conv
from ProtoSeg.lib_PMSAD.models.modules.contrast import momentum_update, l2_normalize, ProjectionHead
from ProtoSeg.lib_PMSAD.models.modules.sinkhorn import distributed_sinkhorn
from timm.models.layers import trunc_normal_
from einops import rearrange, repeat
from torch.nn import Softmax


class HRNet_W48(nn.Module):
    """
    deep high-resolution representation learning for human pose estimation, CVPR2019
    """

    def __init__(self, configer):
        super(HRNet_W48, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        in_channels = 720  # 48 + 96 + 192 + 384
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(in_channels, bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout2d(0.10),
            nn.Conv2d(in_channels, self.num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, x_):
        x = self.backbone(x_)
        _, _, h, w = x[0].size()

        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(x[2], size=(h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(x[3], size=(h, w), mode="bilinear", align_corners=True)

        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        out = self.cls_head(feats)
        out = F.interpolate(out, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        return out


### Basic

class MSAIMModule(nn.Module):
    def __init__(self):
        super(MSAIMModule, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=4),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 4, stride=4),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 4, stride=4),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(3*64*8*8, 128),
            nn.BatchNorm1d(128),
            nn.Tanh()
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Tanh()
        )

        self.fc_a1_a3 = nn.Sequential(
            nn.Linear(64, 6),
            nn.ReLU()  
        )
        self.fc_s1_s2 = nn.Sequential(
            nn.Linear(64, 6),
            nn.Sigmoid()  
        )

        self.cuda()  
        
    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(-1, 1, h, w)  # (b*3, 1, h, w)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(b, -1, 64*8*8)  
        x = torch.flatten(x, start_dim=1) 
  
        
        x = self.fc1(x)
        x = self.fc2(x)

        a1_a3 = self.fc_a1_a3(x)
        s1_s2 = self.fc_s1_s2(x)

        a1_a3 = a1_a3.view(b, -1, 2)
        s1_s2 = s1_s2.view(b, -1, 2)

        a1 = a1_a3[:, :, 0].unsqueeze(-1)
        a3 = a1_a3[:, :, 1].unsqueeze(-1) 
        s1 = s1_s2[:, :, 0].unsqueeze(-1)
        s2 = s1_s2[:, :, 1].unsqueeze(-1)

        return a1, a3, s1, s2
    
    def transform(self, x, a1, a3, s1, s2):

            b, c, h, w = x.shape

            x = x / 255.0  # 归一化到[0,1]

            s1 = s1.unsqueeze(-1)  # (b, 3, 1, 1) 
            s2 = s2.unsqueeze(-1)  # (b, 3, 1, 1) 
            a1 = a1.unsqueeze(-1)  # (b, 3, 1, 1) 
            a3 = a3.unsqueeze(-1)  # (b, 3, 1, 1)
                
            mask1 = (0 <=x) * (x < s1)
            mask2 = (s1 <= x) * (x <= (s1 + s2))
            mask3 = ((s1 + s2) < x) * (x <= 1)

            x1 = a1*x
            x2 = ((a3*(s1+s2-1)-a1*s1+1)/s2)*x + ((a1-a3)*(s1+s2)*s1 + (a3-1)*s1)/s2
            x3 = a3*x-a3+1

            x = x1*mask1 + x2*mask2 + x3*mask3
            x = x.view(b, c, h, w)
            
            x = x * 255.0
            return x
    

class Proto_seg_MSDA(nn.Module):
    def __init__(self, configer):
        super(Proto_seg_MSDA, self).__init__()
        self.configer = configer
        self.gamma = self.configer.get('protoseg', 'gamma')
        self.num_prototype = self.configer.get('protoseg', 'num_prototype')
        self.use_prototype = self.configer.get('protoseg', 'use_prototype')
        self.update_prototype = self.configer.get('protoseg', 'update_prototype')
        self.pretrain_prototype = self.configer.get('protoseg', 'pretrain_prototype')
        self.num_classes = self.configer.get('data', 'num_classes')

        self.msda1 = MSDA(48)
        self.msda2 = MSDA(96)
        self.msda3 = MSDA(192)
        self.msda4 = MSDA(384)

        in_channels = 720
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(in_channels, bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout2d(0.10)
        )
        self.prototypes = nn.Parameter(torch.zeros(self.num_classes, self.num_prototype, in_channels),
                                       requires_grad=True)
        self.proj_head = ProjectionHead(in_channels, in_channels)
        self.feat_norm = nn.LayerNorm(in_channels)
        self.mask_norm = nn.LayerNorm(self.num_classes)
        trunc_normal_(self.prototypes, std=0.02)

    def prototype_learning(self, _c, out_seg, gt_seg, masks):
        pred_seg = torch.max(out_seg, 1)[1]
        mask = (gt_seg == pred_seg.view(-1))

        cosine_similarity = torch.mm(_c, self.prototypes.view(-1, self.prototypes.shape[-1]).t())

        proto_logits = cosine_similarity
        proto_target = gt_seg.clone().float()


        protos = self.prototypes.data.clone()
        for k in range(self.num_classes):
            init_q = masks[..., k]
            init_q = init_q[gt_seg == k, ...]
            if init_q.shape[0] == 0:
                continue

            q, indexs = distributed_sinkhorn(init_q)
            m_k = mask[gt_seg == k]
            c_k = _c[gt_seg == k, ...]
            m_k_tile = repeat(m_k, 'n -> n tile', tile=self.num_prototype)
            m_q = q * m_k_tile  # n x self.num_prototype
            c_k_tile = repeat(m_k, 'n -> n tile', tile=c_k.shape[-1])
            c_q = c_k * c_k_tile  # n x embedding_dim
            f = m_q.transpose(0, 1) @ c_q  # self.num_prototype x embedding_dim
            n = torch.sum(m_q, dim=0)

            if torch.sum(n) > 0 and self.update_prototype is True:
                f = F.normalize(f, p=2, dim=-1)
                new_value = momentum_update(old_value=protos[k, n != 0, :], new_value=f[n != 0, :],
                                            momentum=self.gamma, debug=False)
                protos[k, n != 0, :] = new_value
            proto_target[gt_seg == k] = indexs.float() + (self.num_prototype * k)
        self.prototypes = nn.Parameter(l2_normalize(protos),
                                       requires_grad=False)

        if dist.is_available() and dist.is_initialized():
            protos = self.prototypes.data.clone()
            dist.all_reduce(protos.div_(dist.get_world_size()))
            self.prototypes = nn.Parameter(protos, requires_grad=False)

        return proto_logits, proto_target
    
    def forward(self, x, gt_semantic_seg=None, pretrain_prototype=False):
        _, _, h, w = x[0].size()

        feat1 = self.msda1(x[0])
        feat2 = self.msda2(F.interpolate(x[1], size=(h, w), mode="bilinear", align_corners=True))
        feat3 = self.msda3(F.interpolate(x[2], size=(h, w), mode="bilinear", align_corners=True))
        feat4 = self.msda4(F.interpolate(x[3], size=(h, w), mode="bilinear", align_corners=True))
   
        
        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        c = self.cls_head(feats)

        c = self.proj_head(c)
        _c = rearrange(c, 'b c h w -> (b h w) c')
        _c = self.feat_norm(_c)
        _c = l2_normalize(_c)

        self.prototypes.data.copy_(l2_normalize(self.prototypes))

        masks = torch.einsum('nd,kmd->nmk', _c, self.prototypes)

        out_seg = torch.amax(masks, dim=1)
        out_seg = self.mask_norm(out_seg)
        out_seg = rearrange(out_seg, "(b h w) k -> b k h w", b=feats.shape[0], h=feats.shape[2])

        if pretrain_prototype is False and self.use_prototype is True and gt_semantic_seg is not None:
            gt_seg = F.interpolate(gt_semantic_seg.float(), size=feats.size()[2:], mode='nearest').view(-1)
            contrast_logits, contrast_target = self.prototype_learning(_c, out_seg, gt_seg, masks)
            return {'seg': out_seg, 'logits': contrast_logits, 'target': contrast_target}
        return out_seg

class HRNet_W48_Proto_MSDA(nn.Module):
    """
    deep high-resolution representation learning for human pose estimation, CVPR2019
    """

    def __init__(self, configer):
        super(HRNet_W48_Proto_MSDA, self).__init__()
        self.MSAIM = MSAIMModule()
        self.backbone = BackboneSelector(configer).get_backbone()
        self.protoseg = Proto_seg_MSDA(configer)

    def inverse_normalize(self, x_, mean, std, div_value):

        mean = torch.tensor(mean).view(1, 3, 1, 1).to(x_.device)
        std = torch.tensor(std).view(1, 3, 1, 1).to(x_.device)

        x_ = x_ * std
        x_ = x_ + mean
        x_ = x_ * div_value
        return x_
        
    def normalize(self, x_, mean, std, div_value):

        mean = torch.tensor(mean).view(1, 3, 1, 1).to(x_.device)
        std = torch.tensor(std).view(1, 3, 1, 1).to(x_.device)

        x_ = x_ / div_value
        x_ = x_ - mean
        x_ = x_ / std
        return x_

    def forward(self, x_, gt_semantic_seg=None, pretrain_prototype=False, current_epoch=0):

        div_value = 255.0
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        x_inversed = self.inverse_normalize(x_, mean, std, div_value)

        a1, a3, s1, s2 = self.MSAIM(x_inversed)

        if not next(self.MSAIM.parameters()).requires_grad:
            a1 = torch.ones_like(a1)
            a3 = torch.ones_like(a3)

            s1 = torch.full_like(s1, 0.5)
            s2 = torch.full_like(s2, 0.5)

        x_ = self.MSAIM.transform(x_inversed, a1, a3, s1, s2)

        x_ = self.normalize(x_, mean, std, div_value)

        x = self.backbone(x_)
        out_seg = self.protoseg(x, gt_semantic_seg=gt_semantic_seg, pretrain_prototype=pretrain_prototype)

        return out_seg

class MSDA(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(MSDA, self).__init__()

        self.avg_pool_c = nn.AdaptiveAvgPool2d(1)
        self.max_pool_c = nn.AdaptiveMaxPool2d(1)
        self.mlp_w0 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.mlp_w1 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        self.relu    = nn.ReLU(inplace=True)


        self.branch_convs = nn.ModuleList([
            nn.Conv2d(1, 1, kernel_size=1, padding=0, bias=False), 
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),   
            nn.Conv2d(1, 1, kernel_size=5, padding=2, bias=False),   
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),   
        ])

        self.att_double_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
                nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
            ) for _ in range(4)
        ])
        self.softmax_s = nn.Softmax(dim=1) 


        self.branch_weights = nn.Parameter(torch.ones(4) / 4)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape


        avg_c = self.avg_pool_c(x)   
        max_c = self.max_pool_c(x)
        Gc = self.sigmoid(
            self.mlp_w1(self.relu(self.mlp_w0(avg_c))) +
            self.mlp_w1(self.relu(self.mlp_w0(max_c)))
        )                            

        I_avg_s_map = x.mean(dim=1, keepdim=True)
        I_max_s_map = x.max(dim=1, keepdim=True)[0]


        branch_inputs = [
            self.branch_convs[0](I_avg_s_map),                                    
            self.branch_convs[1](I_avg_s_map),                                    
            self.branch_convs[2](I_avg_s_map),                                    
            self.branch_convs[3](torch.cat([I_avg_s_map, I_max_s_map], dim=1)),   
        ]  

        att_logits = torch.cat(
            [self.att_double_convs[j](branch_inputs[j]) for j in range(4)],
            dim=1
        )                                          
        att_weights = self.softmax_s(att_logits)   


        Vj = self.branch_weights
        gs_sum = sum(
            Vj[j] * (branch_inputs[j] * att_weights[:, j:j+1, :, :])
            for j in range(4)
        )                                          
        G_s_hat = self.sigmoid(gs_sum)             

        return x * Gc * G_s_hat
