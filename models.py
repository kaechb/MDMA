import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weight_norm

from helpers import WeightNormalizedLinear


class BlockGen(nn.Module):
    def __init__(self,embed_dim,hidden,num_heads,):
        super().__init__()
        self.fc0 = nn.Linear(embed_dim, hidden)
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.fc2 = nn.Linear(hidden+embed_dim, embed_dim)
        self.fc1_cls = nn.Linear(hidden+1, embed_dim)
        self.attn = nn.MultiheadAttention(hidden, num_heads, batch_first=True)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(hidden)

    def forward(self,x,x_cls,mask,weight=False):

        res = x.clone()
        x=self.act(self.fc0(x))
        x_cls = self.fc0(x_cls)
        x_cls=self.act(self.ln(x_cls))
        if weight:
            x_cls,w = self.attn(x_cls, x, x, key_padding_mask=mask)
        else:
            x_cls = self.attn(x_cls, x, x, key_padding_mask=mask)[0]
            w=None
        x_cls = self.fc1_cls(torch.cat((x_cls,mask.sum(1).unsqueeze(1).unsqueeze(1)/70),dim=-1))
        x=self.act(self.fc2(torch.cat((x,x_cls.expand(-1,x.shape[1],-1)),dim=-1)))
        x = self.act(self.fc1(x)+res)
        return x,x_cls,w


class Gen(nn.Module):
    def __init__(self, n_dim, l_dim_gen, hidden_gen, num_layers_gen, heads_gen, **kwargs):
        super().__init__()
        self.embbed = nn.Linear(n_dim, l_dim_gen)
        self.encoder = nn.ModuleList([BlockGen(embed_dim=l_dim_gen, num_heads=heads_gen,hidden=hidden_gen) for i in range(num_layers_gen)])
        self.out = WeightNormalizedLinear(l_dim_gen, n_dim)
        self.act = nn.LeakyReLU()

    def forward(self, x,mask,weight=False):
        x = self.act(self.embbed(x))
        if weight:
            ws=[]
        x_cls = x.sum(1).unsqueeze(1).clone()/70#normalize by 70 so it is in the right ballpark
        for layer in self.encoder:
            x, x_cls, w = layer(x,x_cls=x_cls,mask=mask,weight=weight)
            if weight:
                ws.append(w)
        if weight:
            return self.out(x),ws
        else:
            return self.out(x)

class BlockCls(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden,dropout):
        super().__init__()

        self.fc0 = (WeightNormalizedLinear(embed_dim, hidden))
        self.fc1 = (WeightNormalizedLinear(hidden+embed_dim, embed_dim))
        self.fc1_cls = (WeightNormalizedLinear(hidden+1, embed_dim))
        self.attn = weight_norm(nn.MultiheadAttention(hidden, num_heads, batch_first=True, dropout=dropout),"in_proj_weight")

        self.act = nn.LeakyReLU()
        self.ln = nn.LayerNorm(hidden)
        self.hidden = hidden

    def forward(self, x, x_cls, mask,weight=False):
        res = x_cls.clone()
        x = self.act(self.fc0(x))
        x_cls = self.act(self.ln(self.fc0(x_cls)))
        if weight:
            x_cls,w = self.attn(x_cls, x, x, key_padding_mask=mask)
        else:
            x_cls = self.attn(x_cls, x, x, key_padding_mask=mask)[0]
            w=None
        x_cls = self.act(self.fc1_cls(torch.cat((x_cls,mask.sum(1).unsqueeze(1).unsqueeze(1)/70),dim=-1)))
        x=self.act(self.fc1(torch.cat((x,x_cls.expand(-1,x.shape[1],-1)),dim=-1)))
        x_cls =x_cls+res
        return x_cls,x,w


class Disc(nn.Module):
    def __init__(self, n_dim, l_dim, hidden, num_layers, heads,dropout, **kwargs):
        super().__init__()
        self.embbed = WeightNormalizedLinear(n_dim, l_dim)
        self.encoder = nn.ModuleList([BlockCls(embed_dim=l_dim, num_heads=heads, hidden=hidden,dropout=dropout) for i in range(num_layers)])
        self.out = WeightNormalizedLinear(l_dim, 1)
        self.embbed_cls = WeightNormalizedLinear(l_dim+1, l_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, l_dim), requires_grad=True)
        self.act = nn.LeakyReLU()
        self.fc1 = WeightNormalizedLinear(l_dim, hidden)
        self.fc2 = WeightNormalizedLinear(hidden, l_dim)
        self.ln = nn.LayerNorm(l_dim)



    def forward(self, x, mask,weight=False):#mean_field=False
        ws=[]
        x = self.act(self.embbed(x))
        x_cls = torch.cat((x.mean(1).unsqueeze(1).clone(),mask.sum(1).unsqueeze(1).unsqueeze(1)/70),dim=-1)# self.cls_token.expand(x.size(0), 1, -1)
        x_cls = self.act(self.embbed_cls(x_cls))
        for layer in self.encoder:
            x_cls,x,w = layer(x, x_cls=x_cls, mask=mask,weight=weight)
            res=x_cls.clone()
            x_cls=(self.act(x_cls))
            if weight:
                ws.append(w)
        x_cls = self.act(self.ln(self.fc2(self.act(self.fc1(x_cls)))))
        if weight:
            return self.out(x_cls),res,ws
        else:
            return self.out(x_cls),res


if __name__ == "__main__":
    def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

    z = torch.randn(1000, 150, 3)
    mask = torch.zeros((1000, 150)).bool()
    x_cls = torch.randn(1000, 1, 6)
    model =Gen(n_dim=3, l_dim_gen=16, hidden_gen=32, num_layers_gen=8, heads_gen=8).cuda()
    print(count_parameters(model))
    model = Disc(3, 64, 128, 3, 8,0.1).cuda()
    print(model(z.cuda(),mask.cuda(),weight=True)[3])