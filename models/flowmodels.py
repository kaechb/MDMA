from nflows.transforms import CompositeTransform
from nflows.transforms.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform
from nflows.transforms.coupling import AffineCouplingTransform
import nflows as nf

from nflows.flows import base
from nflows.flows.base import Flow
from nflows.nn import nets
from nflows.utils.torchutils import create_random_binary_mask
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.utils as utils


# Example usage:
# model = YourModel()
# apply_normalization(model, use_spectral_norm=True)  # Apply spectral norm
# apply_normalization(model, use_spectral_norm=False) # Apply weight norm


def create_resnet(in_features, out_features,hidden_features,context_features,network_layers, dropout,batchnorm , **kwargs):
        '''This is the network that outputs the parameters of the invertible transformation
        The only arguments can be the in dimension and the out dimenson, the structure
        of the network is defined over the config which is a class attribute
        Context Features: Amount of features used to condition the flow - in our case
        this is usually the mass
        num_blocks: How many Resnet blocks should be used, one res net block is are 1 input+ 2 layers
        and an additive skip connection from the first to the third'''

        return nets.ResidualNet(
                in_features,
                out_features,
                hidden_features=hidden_features,
                context_features=context_features,
                num_blocks=network_layers,
                activation=torch.nn.functional.relu,
                dropout_probability=dropout,
                use_batch_norm=batchnorm
                    )
class Flow(nn.Module):
    def __init__(self, n_dim, coupling_layers, spline, network_layers, hidden_features, tail_bound, num_bins, batchnorm, context_features,dropout,n_part,**kwargs):
        super().__init__()
        self.flows = []
        for i in range(coupling_layers):
            '''This creates the masks for the coupling layers, particle masks are masks
            created such that each feature particle (eta,phi,pt) is masked together or not'''
            mask=create_random_binary_mask(n_dim*n_part)
            #Here are the coupling layers of the flow. There seem to be 3 choices but actually its more or less only 2
            #The autoregressive one is incredibly slow while sampling which does not work together with the constraint
            if spline=="autoreg":
                self.flows += [MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=n_dim*n_part,
                    num_blocks=network_layers,
                    hidden_features=hidden_features,
                    context_features=context_features,
                    tails='linear',
                    tail_bound=tail_bound,
                    num_bins=num_bins,
                    use_residual_blocks=False,
                    use_batch_norm=batchnorm,
                    activation=torch.nn.functional.relu)]

            elif spline:

                    self.flows += [PiecewiseRationalQuadraticCouplingTransform(
                        mask=mask,
                      transform_net_create_fn=lambda x,y: create_resnet(x,y,hidden_features=hidden_features,context_features=context_features,network_layers=network_layers,dropout=dropout,batchnorm=batchnorm),
                        tails='linear',
                        tail_bound=tail_bound,
                        num_bins=num_bins)]

            else:
                self.flows+=[ AffineCouplingTransform(
                    mask=mask,
                    transform_net_create_fn=self.create_resnet)]
        #This sets the distribution in the latent space on which we want to morph onto
        self.q0 = nf.distributions.normal.StandardNormal([n_dim*n_part])

        #Creates working flow model from the list of layer modules
        self.flows=CompositeTransform(self.flows)
        # Construct flow model

        self.flow = base.Flow(distribution=self.q0, transform=self.flows)

class Shape(nn.Module):
    def __init__(self, n_dim,l_dim,num_heads,hidden,num_layers,dropout,context_features, **kwargs) -> None:
        super().__init__()
        self.embbed=nn.Linear(n_dim,l_dim)
        self.enc=nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=l_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden,
                    dropout=dropout,
                    norm_first=False,
                    activation=lambda x: F.leaky_relu(x, 0.2),
                    batch_first=True,
                ),
                num_layers=num_layers,
            )
        self.hidden = nn.Linear(l_dim, context_features)
    def forward(self,x,mask=None):
        return self.hidden(self.enc(self.embbed(x),src_key_padding_mask=mask).sum(1))

class TGen(nn.Module):
    def __init__( self, n_dim, l_dim, hidden, num_layers, num_heads, n_part, dropout, fast=False, norm=False, **kwargs):
        super().__init__()
        self.hidden_nodes = hidden
        self.n_dim = n_dim
        self.l_dim = l_dim
        self.n_part = n_part

        self.embbed = nn.Linear(n_dim, l_dim)

        if not fast:
            self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.l_dim,
                nhead=num_heads,
                dim_feedforward=hidden,
                dropout=dropout,
                norm_first=False,
                activation=lambda x: F.leaky_relu(x, 0.2),
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        else:
            self.encoder=FastTransformerEncoder(self.l_dim,num_heads,hidden,num_layers=num_layers,norm=norm)

        self.dropout = nn.Dropout(dropout / 2)
        self.out2 = nn.Linear(l_dim, n_dim)


    def forward(self, x, mask=None,cond=None):


        x = self.embbed(x)
        x = self.encoder(x, src_key_padding_mask=mask,)#attention_mask.bool()
        x = F.leaky_relu(x)
        x = self.out2(x)
        return x
import torch
import torch.nn as nn

class MDMAAggregation(nn.Module):
    def __init__(self, d_model, hidden, num_heads):
        super().__init__()
        self.num_heads = num_heads

        self.act = nn.LeakyReLU(0.01)
        # Linear layers for query, key, value
        self.attn=nn.MultiheadAttention(hidden, num_heads, batch_first=True)
        self.ln = nn.LayerNorm(hidden)
        self.fc0 = nn.Linear(d_model, hidden)
        self.fc=nn.Linear(2*hidden,d_model)
        self.fc_mf=nn.Linear(hidden,d_model)
    def forward(self, x, x_cls, key_padding_mask=None):
        # Linear projections
        res=x.clone()
        x=self.fc0(self.act(x))
        x_cls=self.ln(self.act(self.fc0(x_cls)))
        mf = self.attn( x_cls, x, x, key_padding_mask=key_padding_mask)[0]
        x=torch.cat((x,mf.expand(-1,x.shape[1],-1).clone()),dim=-1)
        mf = self.act(self.fc_mf(mf))

        x=self.fc(x)+res
        # Handle key padding mask

        return x, mf

# Example usage
class FastTransformerEncoder(nn.Module):
    def __init__(self,d_model, num_heads, hidden, num_layers,norm) -> None:
        super().__init__()
        self.layers = nn.ModuleList([FastTransformerEncoderLayer(d_model,num_heads,hidden,norm=norm) for i in range(num_layers
        )])

    def forward(self, x, src_key_padding_mask=None,return_mf=False):
        x_cls = x.mean(1).unsqueeze(1).clone()
        for layer in self.layers:
            x,x_cls = layer(x, x_cls,src_key_padding_mask=src_key_padding_mask)
        if return_mf:
            return x, x_cls
        return x

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(torch.nn.functional.leaky_relu(self.linear1(x),0.2)))

class FastTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff,norm, dropout=0.):
        super().__init__()
        self.self_attn = MDMAAggregation(d_model,hidden=d_ff, num_heads= num_heads)
        self.norm1 = nn.LayerNorm(d_model) if norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        # self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        # self.norm2 = nn.LayerNorm(d_model)  if norm else nn.Identity()

        # self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, x_cls, src_key_padding_mask=None):
        # Self attention
        src2, x_cls = self.self_attn(src, x_cls,  key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # if self.more:
        #     src2 = self.feed_forward(src)
        #     src = src + self.dropout2(src2)
        #     src = self.norm2(src)
        # # Feed-forward
        # src2 = self.feed_forward(src)
        # src = src + self.dropout2(src2)
        # src = self.norm2(src)
        return src, x_cls

# Example usage

class TDisc(nn.Module):
    def __init__(self,n_dim,l_dim,hidden,num_layers,num_heads,n_part,dropout,fast=False,norm=False,spectralnorm=False,weightnorm=False,**kwargs):
        super().__init__()
        self.hidden_nodes = hidden
        self.n_dim = n_dim
        #         l_dim=n_dim
        self.l_dim = l_dim
        self.n_part = n_part

        self.embbed = nn.Linear(n_dim, l_dim)
        self.fast=fast
        if not fast:
            self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.l_dim,
                nhead=num_heads,
                dim_feedforward=hidden,
                dropout=dropout,
                norm_first=False,
                activation=lambda x: F.leaky_relu(x, 0.2),
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        else:
            self.encoder=FastTransformerEncoder(self.l_dim,num_heads,hidden,num_layers=num_layers,norm=norm)

        self.hidden = nn.Linear(l_dim , 2 * hidden)
        self.hidden2 = nn.Linear(2 * hidden, hidden)
        self.out = nn.Linear(hidden, 1)



    def forward(self, x, m=None, mask=None,cond=None,return_mf=False):


        x = self.embbed(x)
        if self.fast:
            x, mf = self.encoder(x, src_key_padding_mask=mask,return_mf=True)
            x=mf.squeeze(1)
        else:
            x = self.encoder(x, src_key_padding_mask=mask)
            x = torch.sum(x, axis=1)
        x = F.leaky_relu(self.hidden(x), 0.2)
        x = F.leaky_relu(self.hidden2(x), 0.2)
        x = self.out(x)
        if return_mf and self.fast:
            return x, mf
        return x, None

if __name__=="__main__":
    nf=Flow(3,3,True,3,3,0.9,3,True,0,0)
    print(nf.flow.log_prob(torch.randn(100,3)))
