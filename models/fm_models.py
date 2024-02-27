import math

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

class CosineEncoding:
    """Cosine encoding of a tensor.

    adapted from: https://github.com/rodem-hep/PC-JeDi/blob/main/src/models/modules.py
    """

    def __init__(
        self,
        outp_dim: int = 32,
        min_value: float = 0.0,
        max_value: float = 1.0,
        frequency_scaling: str = "exponential",
    ) -> None:
        self.outp_dim = outp_dim
        self.min_value = min_value
        self.max_value = max_value
        self.frequency_scaling = frequency_scaling

    def __call__(self, inpt: torch.Tensor) -> torch.Tensor:
        return cosine_encoding(
            inpt, self.outp_dim, self.min_value, self.max_value, self.frequency_scaling
        )


def cosine_encoding(
    x: torch.Tensor,
    outp_dim: int = 32,
    min_value: float = 0.0,
    max_value: float = 1.0,
    frequency_scaling: str = "exponential",
) -> torch.Tensor:
    """Computes a positional cosine encodings with an increasing series of frequencies.

    The frequencies either increase linearly or exponentially (default).
    The latter is good for when max_value is large and extremely high sensitivity to the
    input is required.
    If inputs greater than the max value are provided, the outputs become degenerate.
    If inputs smaller than the min value are provided, the inputs the the cosine will
    be both positive and negative, which may lead degenerate outputs.

    Always make sure that the min and max bounds are not exceeded!

    Args:
        x: The input, the final dimension is encoded. If 1D then it will be unqueezed
        out_dim: The dimension of the output encoding
        min_value: Added to x (and max) as cosine embedding works with positive inputs
        max_value: The maximum expected value, sets the scale of the lowest frequency
        frequency_scaling: Either 'linear' or 'exponential'

    Returns:
        The cosine embeddings of the input using (out_dim) many frequencies
    """

    # Unsqueeze if final dimension is flat
    if x.shape[-1] != 1 or x.dim() == 1:
        x = x.unsqueeze(-1)

    # Check the the bounds are obeyed
    if torch.any(x > max_value):
        print("Warning! Passing values to cosine_encoding encoding that exceed max!")
    if torch.any(x < min_value):
        print("Warning! Passing values to cosine_encoding encoding below min!")

    # Calculate the various frequencies
    if frequency_scaling == "exponential":
        freqs = torch.arange(outp_dim, device=x.device).exp()
    elif frequency_scaling == "linear":
        freqs = torch.arange(1, outp_dim + 1, device=x.device)
    else:
        raise RuntimeError(f"Unrecognised frequency scaling: {frequency_scaling}")

    return torch.cos((x + min_value) * freqs * math.pi / (max_value + min_value))

def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element. These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden,cond_dim=0,frequencies=8):
        super().__init__()

        self.fc0 =  nn.Linear(embed_dim+2*frequencies, hidden)
        # self.fc0_cls = nn.Linear(embed_dim+1, hidden)
        # self.fc0_cls = nn.Linear(embed_dim+1, hidden)
        self.fc1 = nn.Linear(hidden +embed_dim, embed_dim)
        self.glu = False
        if cond_dim>1:
            self.cond_cls = nn.Linear(cond_dim,hidden)# embed_dim if cond_dim == 1 else
            self.fc1_cls = nn.Linear(hidden , embed_dim)
        else:
            self.fc1_cls = nn.Linear(hidden +cond_dim, embed_dim)
        # self.fc2_cls = nn.Linear(embed_dim, embed_dim)
# embed_dim if cond_dim == 1 else
        self.attn = nn.MultiheadAttention(
                hidden,
                num_heads,
                batch_first=True,
            )

        self.act = nn.LeakyReLU()
        self.ln = nn.LayerNorm(hidden)
        # self.t_local_cat=t_local_cat
        self.cond_dim=cond_dim

    def forward(self, x, x_cls, cond, mask,t=None):
        res = x.clone()


        #res_cls = x_cls.clone()

        x=torch.cat((self.act(x),t.expand(-1,x.shape[1],-1)),dim=-1)

        x_cls=torch.cat((self.act(x_cls),t[:,:1,:]),dim=-1)

        x = self.fc0(x)
        x_cls = self.ln(self.fc0(x_cls))
        x_cls, w = self.attn(x_cls, x, x, key_padding_mask=~mask.squeeze(-1), need_weights=False)
        if self.cond_dim>1:
            x_cls = F.glu(torch.cat((x_cls,self.cond_cls(cond)),dim=-1))
            x_cls = self.fc1_cls(x_cls)
        else:
            x_cls =  self.fc1_cls(torch.cat((x_cls,cond),dim=-1))


#+res_cls
        x = self.fc1(torch.cat((x, x_cls.expand(-1, x.shape[1], -1)), dim=-1)) + res
        return x, x_cls, w


class MDMA(nn.Module):
    def __init__(
        self,
        latent: int ,
        n_dim: int ,
        hidden_dim: int ,
        layers: int ,
        cond_dim: int ,
        frequencies: int ,
        avg_n: int ,
        num_heads: int ,
        **kwargs
    ):
        self.embed = CosineEncoding(
                outp_dim=2 * frequencies,
                min_value=0.0,
                max_value=1.0,
                frequency_scaling="exponential",
            )
        super().__init__()
        self.embbed = nn.Linear(n_dim+2*frequencies, latent)
        self.embbed_cls=nn.Linear(latent+cond_dim ,latent)
        self.encoder = nn.ModuleList([Block(embed_dim=latent, num_heads=num_heads, hidden=hidden_dim,cond_dim=cond_dim,frequencies=frequencies) for i in range(layers)])
        self.out = nn.Linear(latent, n_dim)
        self.act = nn.LeakyReLU()
        self.avg_n=avg_n
        # self.time_embed = nn.Sequential(
        #     nn.Linear(4, 4),
        #     nn.SiLU(),
        #     nn.Linear(4, 1),
        # )
        self.cond = nn.Linear(cond_dim, latent)
        self.cond_dim=cond_dim

    def forward(self,        t: torch.Tensor = None,
        x: torch.Tensor = None,

        mask: torch.Tensor = None,cond=None):
        mask=~mask
        timesteps = t
        while timesteps.dim() > 1:
            timesteps = timesteps[:, 0]
        if timesteps.dim() == 0 or len(timesteps)==1:
            timesteps = timesteps.repeat(x.shape[0])
        t = self.embed(timesteps).unsqueeze(1)

        x=torch.cat((x,t.expand(-1,x.shape[1],-1)),dim=-1)
        x = self.act(self.embbed(x))
        x=x*mask.float().unsqueeze(-1)
        x_cls = x.sum(1).unsqueeze(1).clone() / self.avg_n
        x_cls= torch.cat((x_cls,cond),dim=-1)
        x_cls=self.embbed_cls(x_cls)

        if self.cond_dim>1:
            x_cls = F.glu(torch.cat((x_cls,self.cond(cond.float())),dim=-1))
        else:

            cond=mask.sum(1,keepdim=True).reshape(-1,1,1)

        for layer in self.encoder:
            x, x_cls, w = layer(x, x_cls=x_cls, mask=mask, cond=cond, t=t)
        x = self.out(self.act(x))

        return x*mask.unsqueeze(-1)