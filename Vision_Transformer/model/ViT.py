from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
        To implement the drop path operation in stochastic depth (random depth), 
        it is commonly used in RESNET or vision transformer and other networks. 
        During training, some paths are randomly "discarded" to play a role in regularization and reduce overfitting.

        During training, the characteristic maps of some samples are randomly set to zero with probability drop_prob, 
        and the remaining values are scaled to prevent overfitting.
    """
    if drop_prob == 0. or not training:
        return x
    # calculate keep probably
    keep_prob = 1 - drop_prob
    # generate a mask that has the same shape as x, 
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    # generate mask, if result >= 1 keep, else drop
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    # 1. scaled to prevent overfitting
    # 2. randomly set to zero
    output = x.div(keep_prob) * random_tensor

    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob
        # self.training is a built-in properties of nn.Module

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    "2D image to patch embedding"
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768, norm_layer=None):
        super().__init__()
        self.image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        self.grid_size = (self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.projector = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.image_size[0] and W == self.image_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."

        # flatten: [B, C, H, W] --> [B, C, HW]
        # transpose: [B, C, HW] --> [B, HW, C]
        x = self.projector(x).flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, 
                 dim, 
                 num_heads=8, 
                 qkv_bias=False, 
                 qk_scale=None,      # denominator of qk, the default is (dim ** -0.5).
                 attn_drop_ratio=0., # dropout rate of attention
                 proj_drop_ratio=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_scale = qk_scale if qk_scale else self.head_dim ** -0.5
        # qkv linear
        self.qkv = nn.Linear(in_features=dim, out_features=3 * dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        # fusion projection linear
        self.proj = nn.Linear(in_features=dim, out_features=dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        B, N, C = x.shape

        # generate qkv
        # input:[batch_size, num_patches + 1, total_embed_dim]
        # qkv: [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: [batch_size, num_patches + 1, 3, num_head, embed_dim_per_head]
        # permute: [3, batch_size, num_head, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # [batch_size, num_head, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # calculate multi-head self-attention
        # attention --> softmax --> dropout
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.qk_scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # calculate multi-head value
        # x.shape: [batch_size, num_head, num_patches + 1, embed_dim_per_head]
        # transpose: [batch_size, num_patches + 1, num_head, embed_dim_per_head]
        # reshape: [batch_size, num_patches + 1, total_embed_dim]
        x = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)

        # fuse multi-head value
        # proj --> dropout
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.out_features = out_features or in_features
        self.hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features=in_features, out_features=self.hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(in_features=self.hidden_features, out_features=self.out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))

        return x
    

class Block(nn.Module):
    def __init__(self,
                 dim, 
                 num_heads, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_ratio=0., 
                 attn_drop_ratio=0., 
                 drop_path_ratio=0., 
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadSelfAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                           attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.drop_path = DropPath(drop_prob=drop_path_ratio) if drop_path_ratio > 0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class VisionTransformer(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, num_classes=1000, 
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, 
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0., 
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None, 
                 act_layer=None):
        """
        Args:
            image_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_channels (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        self.norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(image_size=image_size, patch_size=patch_size, in_channels=in_channels, embed_dim=self.embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.parameter.Parameter(torch.zeros([1, 1, self.embed_dim]))
        self.dist_token = nn.parameter.Parameter(torch.zeros([1, 1, self.embed_dim])) if distilled else None
        self.pos_embed = nn.parameter.Parameter(torch.zeros([1, self.num_patches + self.num_tokens, self.embed_dim]))
        self.pos_drop = nn.Dropout(drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=self.embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=self.norm_layer, act_layer=self.act_layer)
            for i in range(depth)
        ])
        self.norm = self.norm_layer(self.embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(in_features=embed_dim, out_features=representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(in_features=self.num_features, out_features=self.num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(in_features=self.embed_dim, out_features=self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        # [B, C, H, W] --> [B, num_patches, embed_dim]
        x = self.patch_embed(x)
        # [1, 1, embed_dim] --> [B, 1, embed_dim]
        cls_token = self.cls_token.expand([x.shape[0], -1, -1])
        # concate
        if self.dist_token is None:
            x = torch.cat([cls_token, x], dim=1) # [B, num_patches + 1, embed_dim]
        else:
            x = torch.cat([cls_token, self.dist_token.expand([x.shape[0], -1, -1]), x], dim=1) # [B, num_patches + 2, embed_dim]
        
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x

def _init_vit_weights(m):
    """
        ViT weight initialization
        :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def vit_base_patch16_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    """
    model = VisionTransformer(image_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    model = VisionTransformer(image_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1hCv0U8pQomwAtHBYc4hmZg  密码: s5hl
    """
    model = VisionTransformer(image_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth
    """
    model = VisionTransformer(image_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224(num_classes: int = 1000):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1cxBgZJJ6qUWPSBNcE4TdRQ  密码: qqt8
    """
    model = VisionTransformer(image_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth
    """
    model = VisionTransformer(image_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth
    """
    model = VisionTransformer(image_size=224,
                              patch_size=32,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_huge_patch14_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: converted weights not currently available, too large for github release hosting.
    """
    model = VisionTransformer(image_size=224,
                              patch_size=14,
                              embed_dim=1280,
                              depth=32,
                              num_heads=16,
                              representation_size=1280 if has_logits else None,
                              num_classes=num_classes)
    return model