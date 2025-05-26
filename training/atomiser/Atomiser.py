from .utils import*
from .nn_comp import*
from .encoding import*
import matplotlib.pyplot as plt


import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Reduce

import time



def cache_fn(f):
    cache = dict()
    @wraps(f)
    def cached_fn(*args, _cache = True, key = None, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if key in cache:
            return cache[key]
        result = f(*args, **kwargs)
        cache[key] = result
        return result
    return cached_fn


def pruning(tokens, attention_mask, percent):
    N = tokens.size(1)
    n_mask = int(N * percent/100.)
    perm = torch.randperm(N, device=tokens.device)
    keep_idx = perm[n_mask:]              
    return tokens[:, keep_idx], attention_mask[:, keep_idx], keep_idx



class Atomiser(pl.LightningModule):
    def __init__(
        self,
        *,
        config,
        transform,
        depth: int,
        input_axis: int = 2,
        num_latents: int = 512,
        latent_dim: int = 512,
        cross_heads: int = 1,
        latent_heads: int = 8,
        cross_dim_head: int = 64,
        latent_dim_head: int = 64,
        num_classes: int = 1000,
        latent_attn_depth: int = 0,
        attn_dropout: float = 0.,
        ff_dropout: float = 0.,
        weight_tie_layers: bool = False,
        self_per_cross_attn: int = 1,
        final_classifier_head: bool = True,
        masking: float = 0.,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['transform'])
        self.input_axis = input_axis
        self.masking = masking
        self.config = config
        self.transform = transform

        # Compute input dim from encodings
        
        dx = self.get_shape_attributes_config("pos")
        dy = self.get_shape_attributes_config("pos")
        dw = self.get_shape_attributes_config("wavelength")
        db = self.get_shape_attributes_config("bandvalue")
        #ok
        input_dim = dx + dy + dw + db

        # Initialize spectral params
        #self.VV = nn.Parameter(torch.empty(dw))
        #self.VH = nn.Parameter(torch.empty(dw))
        #nn.init.trunc_normal_(self.VV, std=0.02, a=-2., b=2.)
        #nn.init.trunc_normal_(self.VH, std=0.02, a=-2., b=2.)

        # Latents
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        nn.init.trunc_normal_(self.latents, std=0.02, a=-2., b=2.)

        get_cross_attn = cache_fn(lambda: PreNorm(
            latent_dim,
            CrossAttention(
                query_dim   = latent_dim,
                context_dim = input_dim,
                heads       = cross_heads,
                dim_head    = cross_dim_head,
                dropout     = attn_dropout,
                use_flash   = True
            ),
            context_dim = input_dim
        ))

        get_cross_ff = cache_fn(lambda: PreNorm(
            latent_dim,
            FeedForward(latent_dim, dropout=ff_dropout)
        ))

        get_latent_attn = cache_fn(lambda: PreNorm(
            latent_dim,
            SelfAttention(
                dim        = latent_dim,
                heads      = latent_heads,
                dim_head   = latent_dim_head,
                dropout    = attn_dropout,
                use_flash  = True
            )
        ))

        get_latent_ff = cache_fn(lambda: PreNorm(
            latent_dim,
            FeedForward(latent_dim, dropout=ff_dropout)
        ))
        #d
        # Build cross/self-attn layers
        self.layers = nn.ModuleList()
        for i in range(depth):
            cache_args = {'_cache': (i>0 and weight_tie_layers)}
            # cross
            cross_attn = get_cross_attn(**cache_args)
            cross_ff   = get_cross_ff(**cache_args)
            # self
            self_attns = nn.ModuleList()
            
            for j in range(self_per_cross_attn):
                self_attns.append(nn.ModuleList([
                    get_latent_attn(**cache_args, key = j),
                    get_latent_ff(**cache_args, key = j)
                ]))

            self.layers.append(nn.ModuleList([cross_attn, cross_ff, self_attns]))

  

        # Classifier
        if final_classifier_head:
            self.to_logits = nn.Sequential(
                LatentAttentionPooling(latent_dim, heads=latent_heads, dim_head=latent_dim_head, dropout=attn_dropout),
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, num_classes)
            )
        else:
            self.to_logits = nn.Identity()
       


    def get_shape_attributes_config(self,attribute):
        if self.config["Atomiser"][attribute+"_encoding"]=="NOPE":
            return 0
        if self.config["Atomiser"][attribute+"_encoding"]=="NATURAL":
            return 1
        if self.config["Atomiser"][attribute+"_encoding"]=="FF":
            if self.config["Atomiser"][attribute+"_num_freq_bands"]==-1:
                return int(self.config["Atomiser"][attribute+"_max_freq"])*2+1
            else:
                return int(self.config["Atomiser"][attribute+"_num_freq_bands"])*2+1
        
        if self.config["Atomiser"][attribute+"_encoding"]=="GAUSSIANS":
            return int(len(self.config["wavelengths_encoding"].keys()))
        

    
                
    




    def forward(self, data, mask=None,resolution=None, training=True):
        # Preprocess tokens + mask
        
        if len(data.shape)==3:
            tokens=data
            tokens_mask=mask
        else:
            tokens, tokens_mask = self.transform.process_data(data, mask,resolution)
        



        b = tokens.shape[0]
        # initialize latents
        x = repeat(self.latents, 'n d -> b n d', b=b)
        # apply mask to tokens
        tokens_mask = tokens_mask.to(torch.bool)
        tokens = tokens.masked_fill_(~tokens_mask.unsqueeze(-1), 0.)

        # cross & self layers
        for (cross_attn, cross_ff, self_attns) in self.layers:
            # optionally prune
            t, m = tokens, tokens_mask
            if self.masking > 0 and training:
                t, m, idx = pruning(t, m, self.masking)
            # cross-attn
            x = cross_attn(x, context=t, mask=m) + x
            x = cross_ff(x) + x
            # restore tokens if pruned
            if self.masking > 0 and training:
                tokens[:, idx] = t
                tokens_mask[:, idx] = m
            # self-attn blocks
            for (sa, ff) in self_attns:
                x = sa(x) + x
                x = ff(x) + x


        # classifier
        return self.to_logits(x)
