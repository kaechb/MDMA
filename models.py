import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm, weight_norm

from helpers import MultiheadL2Attention, WeightNormalizedLinear


class CloudNorm(nn.Module):
    def __init__(self, eps=1e-5):
        super(CloudNorm, self).__init__()
        self.eps = eps

    def forward(self, point_cloud, mask=None):
        """
        :param point_cloud: Tensor of shape (B, N, D)
        :param mask: Boolean tensor of the same shape as point_cloud indicating valid points
        :return: Normalized point cloud
        """
        if mask is None:
            mask = torch.zeros_like(point_cloud, dtype=torch.bool)
        mask = ~mask
        # Create a masked version of the point cloud
        masked_point_cloud = point_cloud * mask.unsqueeze(-1).float()

        # Calculate mean and std dev only for valid points
        sum_points = masked_point_cloud.sum(dim=1, keepdim=False)
        num_valid_points = mask.float().sum(dim=1, keepdim=False)
        mean = sum_points / num_valid_points.unsqueeze(-1)

        variance = ((point_cloud - mean.unsqueeze(1)) ** 2 * mask.unsqueeze(-1).float()).sum(dim=1, keepdim=True) / num_valid_points.unsqueeze(-1).unsqueeze(-1)
        std = torch.sqrt(variance)

        # Normalize the point cloud
        normalized_point_cloud = (point_cloud - mean.unsqueeze(1)) / (std + self.eps)

        # Zero out the points that are masked
        normalized_point_cloud = normalized_point_cloud * mask.unsqueeze(-1).float()

        return normalized_point_cloud


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden, dropout, weightnorm=True, cond_dim=1, glu=False, cloudnorm=False, critic=True, L2=False):
        super().__init__()
        self.fc0 = WeightNormalizedLinear(embed_dim, hidden) if weightnorm else nn.Linear(embed_dim, hidden)
        self.fc0_cls = (WeightNormalizedLinear(embed_dim, hidden)) if weightnorm else nn.Linear(embed_dim, hidden)
        self.fc1 = (WeightNormalizedLinear(hidden * 2, embed_dim)) if weightnorm else nn.Linear(hidden * 2, embed_dim)
        self.glu = glu
        self.cloudnorm = CloudNorm() if cloudnorm else None
        self.fc1_cls = (WeightNormalizedLinear(hidden + 1, hidden)) if weightnorm else nn.Linear(hidden + 1, hidden)
        self.fc2_cls = (WeightNormalizedLinear(hidden, embed_dim)) if weightnorm else nn.Linear(hidden, embed_dim)
        self.cond_cls = WeightNormalizedLinear(cond_dim, hidden) if weight_norm and cond_dim > 1 else nn.Linear(cond_dim,hidden,bias=L2)# embed_dim if cond_dim == 1 else
        self.attn = (
            MultiheadL2Attention(hidden, num_heads)
            if critic and L2
            else weight_norm(nn.MultiheadAttention(hidden, num_heads, batch_first=True, dropout=dropout), "in_proj_weight")
            if weightnorm
            else nn.MultiheadAttention(
                hidden,
                num_heads,
                batch_first=True,
            )
        )
        self.act = nn.LeakyReLU()
        self.ln = nn.LayerNorm(hidden)

    def forward(self, x, x_cls, cond, mask):
        res = x.clone()
        res_cls = x_cls.clone()
        x = self.fc0(self.act(x))
        if self.cloudnorm:
            x = self.cloudnorm(x, mask)
        x_cls = self.ln(self.fc0(self.act(x_cls)))
        x_cls, w = self.attn(x_cls, x, x, key_padding_mask=mask, need_weights=False)
        x_cls = self.fc1_cls(torch.cat((x_cls, cond[:, :, -1:]), dim=-1))
        if self.glu:
            x_cls = self.act(F.glu(torch.cat((x_cls, self.cond_cls(cond[:, :, :1])), dim=-1)))
        x = self.fc1(torch.cat((x, x_cls.expand(-1, x.shape[1], -1)), dim=-1)) + res
        x_cls = self.act(self.fc2_cls(x_cls))
        return x, x_cls, w


class Gen(nn.Module):
    def __init__(self, n_dim, l_dim_gen, hidden_gen, num_layers_gen, heads_gen, cond_dim, cloudnormgen, glu, L2, **kwargs):
        super().__init__()
        self.embbed = nn.Linear(n_dim, l_dim_gen)
        self.encoder = nn.ModuleList([Block(embed_dim=l_dim_gen, num_heads=heads_gen, hidden=hidden_gen, weightnorm=False, dropout=0, cloudnorm=cloudnormgen, glu=glu, critic=False,L2=L2) for i in range(num_layers_gen)])
        self.out = nn.Linear(l_dim_gen, n_dim)
        self.act = nn.LeakyReLU()
        if cond_dim > 1:
            self.cond = nn.Linear(1, l_dim_gen)
        else:
            self.cond = None

    def forward(self, x, mask, cond, weight=False):
        x = self.act(self.embbed(x))

        if self.cond:
            x = F.glu(torch.cat((x, self.cond(cond[:, :, :1]).expand(-1, x.shape[1], -1).clone()), dim=-1))
        x_cls = x.sum(1).unsqueeze(1).clone() / self.avg_n
        for layer in self.encoder:
            x, x_cls, w = layer(x, x_cls=x_cls, mask=mask, cond=cond)
        x = self.out(self.act(x))
        return x


class Disc(nn.Module):
    def __init__(self, n_dim, l_dim, hidden, num_layers, heads, dropout, cond_dim, weightnorm, cloudnorm, glu,L2, **kwargs):
        super().__init__()
        self.embbed = WeightNormalizedLinear(n_dim, l_dim) if weightnorm else spectral_norm(nn.Linear(n_dim, l_dim))
        self.encoder = nn.ModuleList([Block(embed_dim=l_dim, num_heads=heads, hidden=hidden, dropout=dropout, weightnorm=weightnorm, cloudnorm=cloudnorm, glu=glu, critic=True,L2=L2) for i in range(num_layers)])
        self.out = WeightNormalizedLinear(l_dim, 1) if weightnorm else spectral_norm(nn.Linear(l_dim, 1))
        self.cond = True if cond_dim > 1 else False
        self.embbed_cls = WeightNormalizedLinear(l_dim + self.cond * cond_dim, l_dim) if weightnorm else spectral_norm(nn.Linear(l_dim + self.cond * cond_dim, l_dim))
        self.act = nn.LeakyReLU()
        self.fc1 = WeightNormalizedLinear(l_dim + cond_dim, hidden) if weightnorm else nn.Linear(l_dim + cond_dim, hidden)
        self.fc2 = WeightNormalizedLinear(hidden, l_dim) if weightnorm else nn.Linear(hidden, l_dim)
        self.ln = nn.LayerNorm(l_dim)
        if not weightnorm:
            for module in self.encoder:
                for name, submodule in module.named_children():
                    if isinstance(submodule, nn.Linear):
                        setattr(module, name, spectral_norm(submodule))
                    if isinstance(submodule, nn.MultiheadAttention):
                        setattr(module, name, spectral_norm(submodule, "in_proj_weight"))

    def forward(self, x, mask, cond, weight=False):  # mean_field=False
        x = self.act(self.embbed(x))
        x_cls = torch.cat(((x.sum(1) / self.avg_n).unsqueeze(1).clone(), cond), dim=-1) if self.cond else (x.sum(1) / self.avg_n).unsqueeze(1).clone()
        x_cls = self.act(self.embbed_cls(x_cls))
        for layer in self.encoder:
            x, x_cls, w = layer(x, x_cls=x_cls, mask=mask, cond=cond)
            mean_field = x_cls.clone()
        x_cls = self.act(self.fc2(self.act(self.fc1(self.act(torch.cat((x_cls, cond), dim=-1))))))
        return self.out(x_cls), mean_field


if __name__ == "__main__":
    # def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

    z = torch.randn(10, 40, 3, device="cuda")
    mask = torch.zeros((10, 40), device="cuda").bool()
    cond = torch.cat(((~mask.bool()).float().sum(1).unsqueeze(-1).unsqueeze(-1) / 40, torch.randn(10, 1, 1, device="cuda")), dim=-1)
    model = Gen(n_dim=3, l_dim_gen=16, hidden_gen=32, num_layers_gen=8, heads_gen=8, dropout=0.0, cond_dim=2, cloudnormgen=True, glu=False).cuda()
    model.avg_n = 40
    x = model(z.cuda(), mask.cuda(), cond.cuda())
    print(x.std())
    model = Disc(n_dim=3, l_dim=16, hidden=64, num_layers=3, heads=4, dropout=0, cond_dim=2, weightnorm=False, cloudnorm=False, glu=False).cuda()
    model.avg_n = 40
    s, s_cls = model(z.cuda(), mask.cuda(), cond=cond)
    print(s.std(), s_cls.std())
    assert (s == s).all()
