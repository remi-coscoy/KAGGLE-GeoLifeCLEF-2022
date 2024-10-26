
# Standard imports
from functools import reduce
import operator
from typing import Any, Callable, Dict, List, NamedTuple, Optional
import math
from collections import OrderedDict
from functools import partial

# External imports
import torch
import torch.nn as nn

from torchvision.models.vision_transformer import Encoder



class VisionTransformer(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        image_size: int,
        input_channels: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.norm_layer = norm_layer


        self.conv_proj = nn.Conv2d(
            in_channels=input_channels, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
        )

        seq_length = (image_size // patch_size) ** 2

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        self.seq_length = seq_length


        # fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
        # nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
        # if self.conv_proj.bias is not None:
        #     nn.init.zeros_(self.conv_proj.bias)


    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        return x




class VIT(nn.Module):
    def __init__(self, cfg, input_img_size, input_tab_size, num_classes):
        super(VIT, self).__init__()

        self.vit = VisionTransformer(image_size=input_img_size[1], 
                                     input_channels=input_img_size[0]-1+ cfg["embedding_landcover_size"], 
                                     patch_size=cfg["patch_size"], 
                                     num_layers=cfg["num_layers"], 
                                     num_heads=cfg["num_heads"], 
                                     hidden_dim=cfg["hidden_dim"], 
                                     mlp_dim=cfg["mlp_dim"])
        
        self.flat = nn.Flatten(start_dim=1)
        self.lin2 = nn.Linear(num_classes, num_classes)
        
        self.embed = nn.Embedding(34,cfg["embedding_landcover_size"])
        
        self.batchnorm_lin = nn.BatchNorm1d(num_classes)
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(cfg["hidden_dim"] + input_tab_size[0], num_classes)
        self.lin2 = nn.Linear(num_classes,num_classes)
        
        # nn.init.zeros_(self.lin2.weight)
        # nn.init.zeros_(self.lin2.bias)
        # nn.init.zeros_(self.lin1.weight)
        # nn.init.zeros_(self.lin1.bias)
        self.batchnorm_tab = nn.BatchNorm1d(input_tab_size[0])
            
    def forward(self, input_imgs, input_tab):
        imgs = input_imgs[:,:-1,:,:]
        landcover = input_imgs[:,-1,:,:].long()
        landcover = self.embed(landcover)
        landcover = landcover.permute(0,3,1,2)
        x = torch.cat((imgs,landcover),dim=1)
        x = self.vit(x)
        x = self.flat(x)
        input_tab = self.batchnorm_tab(input_tab)
        x = torch.cat((x, input_tab), dim=1)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.batchnorm_lin(x)
        x = self.lin2(x)
        return x