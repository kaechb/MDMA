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
    def __init__(
        self,
        n_dim=3,
        l_dim=10,
        hidden=300,
        num_layers=3,
        num_heads=1,
        n_part=5,
        fc=False,
        dropout=0.5,
        no_hidden=True,
        **kwargs
    ):
        super().__init__()
        self.hidden_nodes = hidden
        self.n_dim = n_dim
        self.l_dim = l_dim
        self.n_part = n_part
        self.no_hidden = no_hidden
        self.fc = fc
        if fc:
            self.l_dim *= n_part
            self.embbed_flat = nn.Linear(n_dim * n_part, l_dim)
            self.flat_hidden = nn.Linear(l_dim, hidden)
            self.flat_hidden2 = nn.Linear(hidden, hidden)
            self.flat_hidden3 = nn.Linear(hidden, hidden)
            self.flat_out = nn.Linear(hidden, n_dim * n_part)
        else:
            self.embbed = nn.Linear(n_dim, l_dim)
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=l_dim,
                    nhead=num_heads,
                    batch_first=True,
                    norm_first=False,
                    dim_feedforward=hidden,
                    dropout=dropout,
                ),
                num_layers=num_layers,
            )
            self.hidden = nn.Linear(l_dim, hidden)
            self.hidden2 = nn.Linear(hidden, hidden)
            self.hidden3 = nn.Linear(hidden, hidden)
            self.dropout = nn.Dropout(dropout / 2)
            self.out = nn.Linear(hidden, n_dim)
            self.out2 = nn.Linear(l_dim, n_dim)

            self.out_flat = nn.Linear(hidden, n_dim * n_part)

    def forward(self, x, mask=None):

        if self.fc:
            x = x.reshape(len(x), self.n_part * self.n_dim)
            x = self.embbed_flat(x)
            x = F.leaky_relu(self.flat_hidden(x))
            #             x = self.dropout(x)
            x = self.flat_out(x)
            x = x.reshape(len(x), self.n_part, self.n_dim)
        else:
            x = self.embbed(x)
            x = self.encoder(x, src_key_padding_mask=mask,)#attention_mask.bool()

            if not self.no_hidden==True:

                x = F.leaky_relu(self.hidden(x))
                x = self.dropout(x)
                x = F.leaky_relu(self.hidden2(x))
                x = self.dropout(x)
                x = self.out(x)
            elif self.no_hidden=="more":
                x = F.leaky_relu(self.hidden(x))
                x = self.dropout(x)
                x = F.leaky_relu(self.hidden2(x))
                x = self.dropout(x)
                x = F.leaky_relu(self.hidden3(x))
                x = self.dropout(x)

            else:
                x = F.leaky_relu(x)
                x = self.out2(x)
        return x

class FastTransformerEncoderLayer(nn.Module):
    def __init__(self,d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="leaky_relu"):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = activation
    def sa_block(self,x,mask=None):
        return torch.nn.functional.scaled_dot_product_attention(x,x,x,mask=mask)
    def ff_block(self,x):
        return self.linear2(self.dropout(F.leaky_relu(self.linear1(x))))
    def forward(self, x, mask=None):
        x=self.norm1(x+self.sa_block(x,mask))
        x=self.norm2(x+self.ff_block(x))


class TDisc(nn.Module):
    def __init__(
        self,
        n_dim=3,
        l_dim=10,
        hidden=300,
        num_layers=3,
        num_heads=1,
        n_part=2,
        fc=False,
        dropout=0.5,
        mass=False,
        clf=False,
        fast=False,
        **kwargs
    ):
        super().__init__()
        self.hidden_nodes = hidden
        self.n_dim = n_dim
        #         l_dim=n_dim
        self.l_dim = l_dim
        self.n_part = n_part
        self.fc = fc
        self.clf = clf

        if fc:
            self.l_dim *= n_part
            self.embbed_flat = nn.Linear(n_dim * n_part, l_dim)
            self.flat_hidden = nn.Linear(l_dim, hidden)
            self.flat_hidden2 = nn.Linear(hidden, hidden)
            self.flat_hidden3 = nn.Linear(hidden, hidden)
            self.flat_out = nn.Linear(hidden, 1)
        else:
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
                self.encoder=nn.Sequential(*[FastTransformerEncoderLayer(self.l_dim,num_heads,hidden,dropout) for i in range(num_layers)])
                
            self.hidden = nn.Linear(l_dim + int(mass), 2 * hidden)
            self.hidden2 = nn.Linear(2 * hidden, hidden)
            self.out = nn.Linear(hidden, 1)

    def forward(self, x, m=None, mask=None):

        if self.fc == True:
            x = x.reshape(len(x), self.n_dim * self.n_part)
            x = self.embbed_flat(x)
            x = F.leaky_relu(self.flat_hidden(x), 0.2)
            x = F.leaky_relu(self.flat_hidden2(x), 0.2)
            x = self.flat_out(x)
        else:
            x = self.embbed(x)
            if self.clf:
                x = torch.concat((torch.ones_like(x[:, 0, :]).reshape(len(x), 1, -1), x), axis=1)
                mask = torch.concat((torch.ones_like((mask[:, 0]).reshape(len(x), 1)), mask), dim=1).to(x.device)

                x = self.encoder(x, src_key_padding_mask=mask)
                x = x[:, 0, :]
            else:
                x = self.encoder(x, src_key_padding_mask=mask)
                x = torch.sum(x, axis=1)
            if m is not None:
                x = torch.concat((m.reshape(len(x), 1), x), axis=1)
            x = F.leaky_relu(self.hidden(x), 0.2)
            x = F.leaky_relu(self.hidden2(x), 0.2)
            x = self.out(x)
            x = x
        return x

if __name__=="__main__":
    nf=Flow(3,3,True,3,3,0.9,3,True,0,0)
    print(nf.flow.log_prob(torch.randn(100,3)))
