import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import random
import utils
import numpy as np
import math
import os
from torch import nn
from torch import Tensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
# from common_spatial_pattern import csp
from torch.autograd import Function
from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


# Convolution module
# use conv to capture local features, instead of postion embedding.
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        # self.patch_size = patch_size
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (64, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth=6, emb_size=40):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])

class ReduceDim(nn.Sequential):
    def __init__(self, emb_size=40):
        super().__init__()
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
        )
        self.reduce_dim = nn.Sequential(
            nn.Linear(1480, 256),
            # nn.ELU(),
            # nn.Dropout(0.5),
        )
    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.reduce_dim(x)
        return out

# projectors
class ProjectorHead(nn.Sequential):
    def __init__(self, mlp_dim=1024, dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(256, mlp_dim, bias=False),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(mlp_dim, affine=False),
            nn.Linear(mlp_dim, dim, bias=False),
            nn.BatchNorm1d(dim, affine=False),  
        )

    def forward(self, x):
        out = self.fc(x)
        return x, out

class ClassificationHead(nn.Sequential):
    def __init__(self, n_classes=4):
        super().__init__()
        self.fc = nn.Sequential(
            # nn.Linear(2440, 256),
            # nn.ELU(),
            # nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes), 
        )

    def forward(self, x):
        out = self.fc(x)
        return out

class Conformer(nn.Sequential):
    def __init__(self, emb_size=40, depth=6, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            TransformerEncoder(depth, emb_size),
            ReduceDim(),
            ProjectorHead()
        )


class CLUDA(nn.Module):
    def __init__(self, number_of_source=103, number_of_category=4, T=0.07, T2=0.03):
        super(CLUDA, self).__init__()
        self.T = T
        self.T2 = T2
        self.number_of_category = number_of_category
        self.number_of_source = number_of_source
        self.encoder = Conformer()
        self.cls = ClassificationHead(number_of_category)

    def forward(self, data_tar=0, data_src=0, label_src=0, number_of_source=103):
        if self.training == True:
            cls_loss = 0
            contrastive_loss = 0
            contrastive_loss_source = 0
            q_list = []
            num_iter = 0
            # compute the pseudo label
            feature_tar, k = self.encoder(data_tar)
            pred_tar = self.cls(feature_tar)
            pred_tar = F.softmax(pred_tar, dim=1)
            pred = pred_tar.data.max(1)[1]
            pred = pred.reshape(-1, 1)
            # compute the contrastive loss
            for i in range(number_of_source):
                feature_q, q = self.encoder(data_src[i])
                feature_qk = torch.cat((q, k))
                labels = torch.cat((label_src[i], pred)).squeeze()
                contrastive_loss += self.inter_subject_contrastive_loss(feature_qk, labels, self.T)
                q_list.append(q)
                # compute the cls loss
                pred_q = self.cls(feature_q)
                cls_loss += F.nll_loss(F.log_softmax(pred_q, dim=1), label_src[i].squeeze())
            cls_loss = cls_loss / number_of_source
            contrastive_loss = contrastive_loss / number_of_source
            # compute contrastive loss between source
            for i in range(len(q_list)):
                for j in range(i+1, len(q_list)):
                    num_iter = num_iter + 1
                    qk = torch.cat((q_list[i], q_list[j]))
                    label_qk = torch.cat((label_src[i], label_src[j]))
                    contrastive_loss_source += self.inter_subject_contrastive_loss(qk, label_qk, self.T2)
            contrastive_loss_source = contrastive_loss_source / num_iter
            return cls_loss, contrastive_loss, contrastive_loss_source
        else:
            feature_tar, _ = self.encoder(data_tar)
            pred_tar = self.cls(feature_tar)
            return pred_tar
    
    # the core code for contrastive learning
    def inter_subject_contrastive_loss(self, feats, labels, tau):
        N = feats.size(0)

        labels = labels.contiguous().view(-1, 1)
        cls_mask = torch.eq(labels, labels.T).float().to(device)

        # the default mask includes all samples but the anchor itself
        # 形状为(N × N)的全 1 矩阵，对角线上的元素全是 0 ，其他全是 1
        default_mask = torch.ones((N, N)).fill_diagonal_(0).to(device)

        # construct the positive set
        positive_set_mask = default_mask * cls_mask

        # construct the anchor set
        anchor_set_mask = default_mask * (1 - cls_mask)

        feats = feats / torch.norm(feats, p=2, dim=1).unsqueeze(1)
        
        sim_matrix = torch.matmul(feats, torch.transpose(feats, 0, 1)) / tau

        sim_matrix_exp = torch.exp(sim_matrix)
        sim_matrix_exp = sim_matrix_exp.clone().fill_diagonal_(0)

        scores = (sim_matrix_exp * positive_set_mask).sum(dim=0).clamp_(1e-6) / ((sim_matrix_exp * anchor_set_mask).sum(dim=0) + (sim_matrix_exp * positive_set_mask).sum(dim=0))
        # scores = (sim_matrix_exp * positive_set_mask).sum(dim=0) / (sim_matrix_exp * anchor_set_mask).sum(dim=0)
        loss_contrast = -torch.log(scores).mean()
        return loss_contrast