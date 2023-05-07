import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class SuperLayerNorm(nn.LayerNorm):
    def __init__(self, embed_dim):
        super().__init__(embed_dim)

        # sampled
        self.sampled_embed_dim = None
        self.sampled_weight = None
        self.sampled_bias = None

    def set_sample_config(self, sampled_embed_dim):
        self.sampled_embed_dim = sampled_embed_dim
        self.sampled_weight = self.weight[:self.sampled_embed_dim]
        self.sampled_bias = self.bias[:self.sampled_embed_dim]

    def forward(self, x):
        return F.layer_norm(x, (self.sampled_embed_dim,), weight=self.sampled_weight, bias=self.sampled_bias, eps=self.eps)

    def params(self):
        params = 0
        params += self.sampled_weight.numel()
        params += self.sampled_bias.numel()
        return params

    def flops(self, N):
        flops = 0
        flops += N * self.sampled_embed_dim * 2
        return flops


class SuperLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, scale=False):
        super().__init__(in_features, out_features, bias=bias)

        self.scale = scale
        # sampled
        self.sampled_in_features = None
        self.sampled_out_features = None
        self.sampled_weight = None
        self.sampled_bias = None
        self.sampled_scale = None

    def set_sample_config(self, sampled_in_features, sampled_out_features):
        self.sampled_in_features = sampled_in_features
        self.sampled_out_features = sampled_out_features
        self.sampled_weight = self.weight[:self.sampled_out_features, :self.sampled_in_features]
        if self.bias is not None:
            self.sampled_bias = self.bias[:self.sampled_out_features]
        if self.scale:
            self.sampled_scale = self.out_features / self.sampled_out_features

    def forward(self, x):
        return F.linear(x, self.sampled_weight, self.sampled_bias) * (self.sampled_scale if self.scale else 1)

    def params(self):
        params = 0
        params += self.sampled_weight.numel()
        if self.sampled_bias is not None:
            params += self.sampled_bias.numel()
        return params

    def flops(self, N):
        flops = 0
        flops += N * np.prod(self.sampled_weight.size())
        if self.sampled_bias is not None:
            flops += N * np.prod(self.sampled_bias.size())
        return flops


class SuperMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., scale=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features or in_features
        self.hidden_features = hidden_features or in_features
        self.fc1 = SuperLinear(self.in_features, self.hidden_features, scale=scale)
        self.act = act_layer()
        self.fc2 = SuperLinear(self.hidden_features, self.out_features, scale=scale)
        self.drop = nn.Dropout(drop)

        self.scale = scale
        # sampled
        self.sampled_in_features = None
        self.sampled_hidden_features = None
        self.sampled_out_features = None
        self.sampled_drop = None

    def set_sample_config(self, sampled_in_features, sampled_hidden_features, sampled_out_features, sampled_drop):
        self.sampled_in_features = sampled_in_features
        self.sampled_hidden_features = sampled_hidden_features or sampled_in_features
        self.sampled_out_features = sampled_out_features or sampled_in_features
        self.sampled_drop = sampled_drop

        self.fc1.set_sample_config(sampled_in_features=self.sampled_in_features, sampled_out_features=self.sampled_hidden_features)
        self.fc2.set_sample_config(sampled_in_features=self.sampled_hidden_features, sampled_out_features=self.sampled_out_features)
        self.drop.p = self.sampled_drop

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def params(self):
        params = 0
        params += self.fc1.params()
        params += self.fc2.params()
        return params

    def flops(self, N):
        flops = 0
        flops += self.fc1.flops(N)
        flops += self.fc2.flops(N)
        return flops


class SuperPatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, scale=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

        self.scale = scale
        # sampled
        self.sampled_embed_dim = None
        self.sampled_weight = None
        self.sampled_bias = None
        self.sampled_scale = None

    def set_sample_config(self, sampled_embed_dim):
        self.sampled_embed_dim = sampled_embed_dim
        self.sampled_weight = self.proj.weight[:sampled_embed_dim, ...]
        if self.proj.bias is not None:
            self.sampled_bias = self.proj.bias[:self.sampled_embed_dim, ...]
        if self.scale:
            self.sampled_scale = self.embed_dim / self.sampled_embed_dim
        if self.norm is not None:
            self.norm.set_sample_config(sampled_embed_dim=self.sampled_embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = F.conv2d(x, self.sampled_weight, self.sampled_bias, stride=self.patch_size, padding=self.proj.padding, dilation=self.proj.dilation).flatten(2).transpose(1, 2) * (self.sampled_scale if self.scale else 1)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def params(self):
        params = 0
        params += self.sampled_weight.numel()
        if self.sampled_bias is not None:
            params += self.sampled_bias.numel()
        if self.norm is not None:
            params += self.norm.params()
        return params

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = 0
        flops += Ho * Wo * np.prod(self.sampled_weight.size())
        if self.sampled_bias is not None:
            flops += Ho * Wo * np.prod(self.sampled_bias.size())
        if self.norm is not None:
            flops += self.norm.flops(Ho * Wo)
        return flops


class SuperPatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=SuperLayerNorm, scale=False):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

        self.scale = scale
        # sampled
        self.sampled_dim = None
        self.sampled_weight = None
        self.sampled_bias = None
        self.sampled_scale = None

    def set_sample_config(self, sampled_dim):
        self.sampled_dim = sampled_dim
        self.sampled_weight = torch.cat([self.reduction.weight[:2 * self.sampled_dim,i:4 * self.sampled_dim:4] for i in range(4)], dim=1)
        if self.reduction.bias is not None:
            self.sampled_bias = self.reduction.bias[:2 * self.sampled_dim]
        if self.scale:
            self.sampled_scale = self.dim / self.sampled_dim
        self.norm.sampled_embed_dim = 4 * self.sampled_dim
        self.norm.sampled_weight = torch.cat([self.norm.weight[i:4 * self.sampled_dim:4] for i in range(4)], dim=0)
        self.norm.sampled_bias = torch.cat([self.norm.bias[i:4 * self.sampled_dim:4] for i in range(4)], dim=0)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = F.linear(x, self.sampled_weight, self.sampled_bias) * (self.sampled_scale if self.scale else 1)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def params(self):
        params = 0
        params += self.norm.params()
        params += self.sampled_weight.numel()
        if self.sampled_bias is not None:
            params += self.sampled_bias.numel()
        return params

    def flops(self):
        H, W = self.input_resolution
        flops = 0
        flops += self.norm.flops(H * W)
        flops += (H // 2) * (W // 2) * np.prod(self.sampled_weight.size())
        if self.sampled_bias is not None:
            flops += (H // 2) * (W // 2) * np.prod(self.sampled_bias.size())
        return flops


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SuperWindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., scale=False):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.qk_dim = num_heads * 32
        self.qk_scale = qk_scale

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1), (2 * window_size[1] - 1), num_heads))  # 2*Wh-1, 2*Ww-1, nH

        self.qkv = nn.Linear(dim, self.qk_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = SuperLinear(self.qk_dim, dim, scale=scale)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        self.scale = scale
        # sampled
        self.sampled_proj_drop = None
        self.sampled_attn_drop = None
        self.sampled_dim = None
        self.sampled_qk_dim = None
        self.sampled_num_heads = None
        self.sampled_window_size = None
        self.sampled_qk_scale = None
        self.sampled_scale = None
        self.sampled_qkv_weight = None
        self.sampled_qkv_bias = None
        self.sampled_relative_position_bias_table = None

    def set_sample_config(self, sampled_dim=None, sampled_qk_dim=None, sampled_num_heads=None, sampled_proj_drop=None, sampled_attn_drop=None, sampled_window_size=None):
        self.sampled_proj_drop = sampled_proj_drop
        self.sampled_attn_drop = sampled_attn_drop
        self.sampled_dim = sampled_dim
        self.sampled_qk_dim = sampled_qk_dim
        self.sampled_num_heads = sampled_num_heads
        self.sampled_window_size = sampled_window_size
        self.sampled_qk_scale = self.qk_scale or (self.sampled_qk_dim // self.sampled_num_heads) ** -0.5

        start = [self.window_size[0] - self.sampled_window_size[0], self.window_size[1] - self.sampled_window_size[1]]
        self.sampled_relative_position_bias_table = self.relative_position_bias_table[start[0]:start[0]+2*self.sampled_window_size[0]-1,start[1]:start[1]+2*self.sampled_window_size[1]-1,:self.sampled_num_heads]
        self.sampled_qkv_weight = torch.cat([self.qkv.weight[i:self.sampled_qk_dim*3:3, :self.sampled_dim] for i in range(3)], dim =0)
        if self.qkv.bias is not None:
            self.sampled_qkv_bias = torch.cat([self.qkv.bias[i:self.sampled_qk_dim*3:3] for i in range(3)], dim =0)
        self.proj.set_sample_config(sampled_in_features=self.sampled_qk_dim, sampled_out_features=self.sampled_dim)
        if self.scale:
            self.sampled_scale = self.dim / self.sampled_qk_dim
        self.attn_drop.p = self.sampled_attn_drop
        self.proj_drop.p = self.sampled_proj_drop

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = F.linear(x, self.sampled_qkv_weight, self.sampled_qkv_bias) * (self.sampled_scale if self.scale else 1)
        qkv=qkv.reshape(B_, N, 3, self.sampled_num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.sampled_qk_scale
        attn = (q @ k.transpose(-2, -1))

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.sampled_window_size[0], device=next(self.parameters()).device)
        coords_w = torch.arange(self.sampled_window_size[1], device=next(self.parameters()).device)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.sampled_window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.sampled_window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.sampled_window_size[1] - 1
        sampled_relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

        relative_position_bias = self.sampled_relative_position_bias_table.reshape(-1, self.sampled_num_heads)[sampled_relative_position_index.view(-1)].reshape(
            self.sampled_window_size[0] * self.sampled_window_size[1], self.sampled_window_size[0] * self.sampled_window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.sampled_num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.sampled_num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, -1) * (self.sampled_scale if self.scale else 1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def params(self):
        params = 0
        params += self.sampled_relative_position_bias_table.numel()
        params += self.sampled_qkv_weight.numel()
        if self.sampled_qkv_bias is not None:
            params += self.sampled_qkv_bias.numel()
        params += self.proj.params()
        return params

    def flops(self, nW, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * np.prod(self.sampled_qkv_weight.size())
        if self.sampled_qkv_bias is not None:
            flops += N * np.prod(self.sampled_qkv_bias.size())
        # attn = (q @ k.transpose(-2, -1))
        flops += self.sampled_num_heads * N * (self.sampled_qk_dim // self.sampled_num_heads) * N
        # attn = attn + relative_position_bias.unsqueeze(0)
        flops += self.sampled_num_heads * N * N
        #  x = (attn @ v)
        flops += self.sampled_num_heads * N * N * (self.sampled_qk_dim // self.sampled_num_heads)
        # x = self.proj(x)
        flops += self.proj.flops(N)
        return flops * nW


class SuperSwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=SuperLayerNorm, scale=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = SuperWindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, scale=scale)

        self.drop_path_rate = drop_path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = SuperMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, scale=scale)

        self.scale = scale
        # sampled
        self.is_identity_layer = None
        self.sampled_dim = None
        self.sampled_num_heads = None
        self.sampled_window_size = None
        self.sampled_mlp_ratio = None
        self.sampled_drop = None
        self.sampled_attn_drop = None
        self.sampled_drop_path = None
        self.sampled_shift_size = None

    def set_sample_config(self, is_identity_layer, sampled_dim=None, sampled_num_heads=None, sampled_window_size=None, sampled_mlp_ratio=None, sampled_drop=None, sampled_attn_drop=None, sampled_drop_path=None):

        if is_identity_layer:
            self.is_identity_layer = True
            return

        self.is_identity_layer = False
        self.sampled_dim = sampled_dim
        self.sampled_num_heads = sampled_num_heads
        self.sampled_window_size = sampled_window_size
        self.sampled_mlp_ratio = sampled_mlp_ratio
        self.sampled_drop = sampled_drop
        self.sampled_attn_drop = sampled_attn_drop
        self.sampled_drop_path = sampled_drop_path
        self.sampled_shift_size = 0 if self.shift_size == 0 else self.sampled_window_size // 2
        assert 0 <= self.sampled_shift_size < self.sampled_window_size, "shift_size must in 0-window_size"

        self.norm1.set_sample_config(sampled_embed_dim=self.sampled_dim)
        self.attn.set_sample_config(sampled_dim=self.sampled_dim, sampled_qk_dim=self.sampled_num_heads*32, sampled_num_heads=self.sampled_num_heads, sampled_proj_drop=self.sampled_drop, sampled_attn_drop=self.sampled_attn_drop, sampled_window_size=to_2tuple(self.sampled_window_size))
        if isinstance(self.drop_path, DropPath):
            if self.sampled_drop_path is not None:
                self.drop_path.drop_prob = self.sampled_drop_path
            else:
                self.drop_path.drop_prob = self.drop_path_rate * self.sampled_dim / self.dim
        self.norm2.set_sample_config(sampled_embed_dim=self.sampled_dim)
        self.sampled_mlp_hidden_dim = int(self.sampled_dim * self.sampled_mlp_ratio)
        self.mlp.set_sample_config(sampled_in_features=self.sampled_dim, sampled_hidden_features=self.sampled_mlp_hidden_dim, sampled_out_features=None, sampled_drop=self.sampled_drop)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        if self.is_identity_layer:
            return x

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.sampled_window_size - W % self.sampled_window_size) % self.sampled_window_size
        pad_b = (self.sampled_window_size - H % self.sampled_window_size) % self.sampled_window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.sampled_shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.sampled_shift_size, -self.sampled_shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.sampled_window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.sampled_window_size * self.sampled_window_size, C)  # nW*B, window_size*window_size, C

        if self.sampled_shift_size > 0:
            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, Hp, Wp, 1), device=next(self.parameters()).device)  # 1 H W 1
            h_slices = (slice(0, -self.sampled_window_size),
                        slice(-self.sampled_window_size, -self.sampled_shift_size),
                        slice(-self.sampled_shift_size, None))
            w_slices = (slice(0, -self.sampled_window_size),
                        slice(-self.sampled_window_size, -self.sampled_shift_size),
                        slice(-self.sampled_shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.sampled_window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.sampled_window_size * self.sampled_window_size)
            sampled_attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            sampled_attn_mask = sampled_attn_mask.masked_fill(sampled_attn_mask != 0, float(-100.0)).masked_fill(sampled_attn_mask == 0, float(0.0))
        else:
            sampled_attn_mask = None

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=sampled_attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.sampled_window_size, self.sampled_window_size, C)
        shifted_x = window_reverse(attn_windows, self.sampled_window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.sampled_shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.sampled_shift_size, self.sampled_shift_size), dims=(1, 2))
        else:
            x = shifted_x
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def params(self):
        params = 0
        if not self.is_identity_layer:
            params += self.norm1.params()
            params += self.attn.params()
            params += self.norm2.params()
            params += self.mlp.params()
        return params

    def flops(self):
        flops = 0
        if not self.is_identity_layer:
            H, W = self.input_resolution
            # norm1
            flops += self.norm1.flops(H * W)
            # W-MSA/SW-MSA
            nW = int(np.ceil(H / self.sampled_window_size) * np.ceil(W / self.sampled_window_size))
            flops += self.attn.flops(nW, self.sampled_window_size * self.sampled_window_size)
            # norm2
            flops += self.norm2.flops(H * W)
            # mlp
            flops += self.mlp.flops(H * W)
        return flops


class SuperBasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=SuperLayerNorm, downsample=None, use_checkpoint=False, scale=False, shift=True):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SuperSwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                      num_heads=num_heads, window_size=window_size,
                                      shift_size=0 if (i % 2 == 0) or (not shift) else window_size // 2,
                                      mlp_ratio=mlp_ratio,
                                      qkv_bias=qkv_bias, qk_scale=qk_scale,
                                      drop=drop, attn_drop=attn_drop,
                                      drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                      norm_layer=norm_layer,
                                      scale=scale)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer, scale=scale)
        else:
            self.downsample = None

        self.scale = scale
        # sampled
        self.sampled_depth = None
        self.sampled_dim = None
        self.sampled_mlp_ratio = None
        self.sampled_num_heads = None
        self.sampled_window_size = None
        self.sampled_drop = None
        self.sampled_attn_drop = None
        self.sampled_drop_path = None

    def set_sample_config(self, sampled_depth, sampled_dim, sampled_num_heads, sampled_window_size, sampled_mlp_ratio, sampled_drop, sampled_attn_drop, sampled_drop_path):
        self.sampled_depth = sampled_depth
        self.sampled_dim = sampled_dim
        self.sampled_mlp_ratio = sampled_mlp_ratio
        self.sampled_num_heads = sampled_num_heads
        self.sampled_window_size = sampled_window_size
        self.sampled_drop = sampled_drop
        self.sampled_attn_drop = sampled_attn_drop
        self.sampled_drop_path = sampled_drop_path
        for i, block in enumerate(self.blocks):
            if i < self.sampled_depth:
                block.set_sample_config(is_identity_layer=False, sampled_dim=self.sampled_dim, sampled_num_heads=self.sampled_num_heads[i], sampled_window_size=self.sampled_window_size[i], sampled_mlp_ratio=self.sampled_mlp_ratio[i], sampled_drop=self.sampled_drop, sampled_attn_drop=self.sampled_attn_drop, sampled_drop_path=self.sampled_drop_path[i] if self.sampled_drop_path is not None else None)
            else:
                block.set_sample_config(is_identity_layer=True)

        if self.downsample is not None:
            self.downsample.set_sample_config(sampled_dim=self.sampled_dim)

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def params(self):
        params = 0
        for blk in self.blocks:
            params += blk.params()
        if self.downsample is not None:
            params += self.downsample.params()
        return params

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class ViTSoup(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=SuperLayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, scale=False, shift=True, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate

        # split image into non-overlapping patches
        self.patch_embed = SuperPatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None, scale=scale)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = SuperBasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                    input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                      patches_resolution[1] // (2 ** i_layer)),
                                    depth=depths[i_layer],
                                    num_heads=num_heads[i_layer],
                                    window_size=window_size,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop_rate, attn_drop=attn_drop_rate,
                                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                    norm_layer=norm_layer,
                                    downsample=SuperPatchMerging if (i_layer < self.num_layers - 1) else None,
                                    use_checkpoint=use_checkpoint,
                                    scale=scale,
                                    shift=shift)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = SuperLinear(self.num_features, num_classes, scale=scale) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

        self.scale = scale
        # sampled
        self.sampled_depths = None
        self.sampled_num_layers = None
        self.sampled_embed_dim = None
        self.sampled_num_features = None
        self.sampled_mlp_ratio = None
        self.sampled_num_heads = None
        self.sampled_window_size = None

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def set_sample_config(self, config):
        self.sampled_depths = config['depth']
        self.sampled_num_layers = len(self.sampled_depths)
        self.sampled_embed_dim = config['embed_dim']
        self.sampled_num_features = int(self.sampled_embed_dim * 2 ** (self.sampled_num_layers - 1))
        self.sampled_mlp_ratio = config['mlp_ratio']
        self.sampled_num_heads = config['num_heads']
        self.sampled_window_size = config['window_size']

        self.patch_embed.set_sample_config(sampled_embed_dim=self.sampled_embed_dim)

        if self.ape:
            self.sampled_absolute_pos_embed = self.absolute_pos_embed[:, :, :self.sampled_embed_dim]

        self.pos_drop.p = self.drop_rate * self.sampled_embed_dim / self.embed_dim

        cum_depths = [0] + np.cumsum(self.sampled_depths).tolist()
        for i_layer in range(self.sampled_num_layers):
            sampled_dim = int(self.sampled_embed_dim * 2 ** i_layer)
            dim = int(self.embed_dim * 2 ** i_layer)
            sampled_drop = self.drop_rate * sampled_dim / dim
            sampled_attn_drop = self.attn_drop_rate * sampled_dim / dim
            self.layers[i_layer].set_sample_config(
                sampled_depth=self.sampled_depths[i_layer],
                sampled_dim=sampled_dim,
                sampled_num_heads=self.sampled_num_heads[cum_depths[i_layer]: cum_depths[i_layer+1]],
                sampled_window_size=self.sampled_window_size[cum_depths[i_layer]: cum_depths[i_layer+1]],
                sampled_mlp_ratio=self.sampled_mlp_ratio[cum_depths[i_layer]: cum_depths[i_layer+1]],
                sampled_drop=sampled_drop,
                sampled_attn_drop=sampled_attn_drop,
                sampled_drop_path=None)

        self.norm.set_sample_config(sampled_embed_dim=self.sampled_num_features)
        if self.num_classes > 0:
            self.head.set_sample_config(sampled_in_features=self.sampled_num_features, sampled_out_features=self.num_classes)

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.sampled_absolute_pos_embed
        x = self.pos_drop(x)

        for i_layer in range(self.sampled_num_layers):
            x = self.layers[i_layer](x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def params(self):
        params = 0
        params += self.patch_embed.params()
        if self.ape:
            params += self.sampled_absolute_pos_embed.numel()
        for i_layer in range(self.sampled_num_layers):
            params += self.layers[i_layer].params()
        params += self.norm.params()
        if self.num_classes > 0:
            params += self.head.params()
        return params

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        if self.ape:
            flops += np.prod(self.sampled_absolute_pos_embed.size())
        for i_layer in range(self.sampled_num_layers):
            flops += self.layers[i_layer].flops()
        flops += self.norm.flops(self.patches_resolution[0] * self.patches_resolution[1] // (2 ** (self.sampled_num_layers-1)) // (2 ** (self.sampled_num_layers-1)))
        if self.num_classes > 0:
            flops += self.head.flops(1)
        return flops
