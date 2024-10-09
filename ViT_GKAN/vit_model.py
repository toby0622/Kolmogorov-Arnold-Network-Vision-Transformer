import torch
import torch.nn as nn
from functools import partial
from collections import OrderedDict


# 定義 GaussianKANLayer 類別
class GaussianKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_gaussians):
        super(GaussianKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.num_gaussians = num_gaussians

        # 初始化高斯函數的中心 (mu) 和標準差 (sigma)
        self.mu = nn.Parameter(torch.empty(input_dim, num_gaussians))
        self.log_sigma = nn.Parameter(
            torch.empty(input_dim, num_gaussians)
        )  # 使用 log_sigma 以保證 sigma 為正

        # 初始化參數
        nn.init.uniform_(self.mu, -1.0, 1.0)  # 根據需要調整範圍
        nn.init.constant_(self.log_sigma, 0.0)  # 初始 sigma 為 1

        # 初始化線性組合的係數
        self.coeffs = nn.Parameter(torch.empty(input_dim, output_dim, num_gaussians))
        nn.init.normal_(self.coeffs, mean=0.0, std=1 / (input_dim * num_gaussians))

    def forward(self, x):
        """
        前向傳播：
        1. 將輸入 x 正規化到 [-1, 1]（根據需要，可以調整或移除此步驟）。
        2. 計算高斯基底函數。
        3. 將基底函數與係數相乘並求和，生成輸出。
        """
        # 將輸入正規化到 [-1, 1]，根據需要可以調整
        x = torch.tanh(x)

        # 將 x 重塑為 (batch_size, inputdim, 1) 以便與 mu 和 sigma 進行廣播運算
        x = x.view(-1, self.inputdim, 1)  # shape: (batch_size, inputdim, 1)

        # 計算 sigma，確保其為正數
        sigma = torch.exp(self.log_sigma) + 1e-8  # 防止除以零

        # 計算高斯基底函數：exp(-((x - mu)/sigma)^2)
        gaussians = torch.exp(
            -(((x - self.mu) / sigma) ** 2)
        )  # shape: (batch_size, inputdim, num_gaussians)

        # 計算線性組合：對每個輸出維度進行加權求和
        # coeffs 的形狀為 (inputdim, output_dim, num_gaussians)
        # 使用愛因斯坦求和約定進行批量矩陣乘法
        y = torch.einsum(
            "bid,iod->bo", gaussians, self.coeffs
        )  # shape: (batch_size, outdim)

        # 重新塑形回原始序列格式
        y = y.view(-1, self.outdim)

        return y


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # 適用於不同維度的張量，而不僅僅是 2D 卷積網絡
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # 二值化
    output = x.div(keep_prob) * random_tensor

    return output


# 每個樣本的 drop path (隨機深度，應用於殘差塊的主路徑時)
class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# 2D 影像 Patch Embedding
class PatchEmbed(nn.Module):
    def __init__(
        self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None
    ):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(
            in_c, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"輸入圖像大小 ({H}*{W}) 與模型大小 ({self.img_size[0]}*{self.img_size[1]}) 不匹配。"

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,  # 輸入 token 的維度
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop_ratio=0.0,
        proj_drop_ratio=0.0,
    ):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# Vision Transformer 使用的 MLP
class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# 修改後的 Block 類別，使用 GaussianKANLayer 取代 ChebyKANLayer
class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_ratio=0.0,
        attn_drop_ratio=0.0,
        drop_path_ratio=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        num_gaussians=5,  # 使用 num_gaussians 取代 degree
    ):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_ratio=attn_drop_ratio,
            proj_drop_ratio=drop_ratio,
        )
        self.drop_path = (
            DropPath(drop_path_ratio) if drop_path_ratio > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        # 使用兩層 GaussianKANLayer 模擬 MLP 結構
        self.gaussiankan1 = GaussianKANLayer(
            input_dim=dim, output_dim=mlp_hidden_dim, num_gaussians=num_gaussians
        )
        self.gaussiankan2 = GaussianKANLayer(
            input_dim=mlp_hidden_dim, output_dim=dim, num_gaussians=num_gaussians
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gaussiankan2(self.gaussiankan1(self.norm2(x))))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_c=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        representation_size=None,
        distilled=False,
        drop_ratio=0.0,
        attn_drop_ratio=0.0,
        drop_path_ratio=0.0,
        embed_layer=PatchEmbed,
        norm_layer=None,
        act_layer=None,
        num_gaussians=5,  # 新增 num_gaussians 參數
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
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
            num_gaussians (int): number of Gaussian functions in GaussianKANLayer
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = (
            embed_dim  # 為了與其他模型一致，使用 num_features
        )
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = (
            nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_tokens, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_ratio, depth)
        ]  # 隨機深度衰減規則
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop_ratio=drop_ratio,
                    attn_drop_ratio=attn_drop_ratio,
                    drop_path_ratio=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    num_gaussians=num_gaussians,  # 傳遞 num_gaussians
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # 表示層
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                OrderedDict(
                    [
                        ("fc", nn.Linear(embed_dim, representation_size)),
                        ("act", nn.Tanh()),
                    ]
                )
            )
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # 分類頭
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.head_dist = None
        if distilled:
            self.head_dist = (
                nn.Linear(self.embed_dim, self.num_classes)
                if num_classes > 0
                else nn.Identity()
            )

        # 權重初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        # [1, 1, embed_dim] -> [B, 1, embed_dim]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, num_patches + 1, embed_dim]
        else:
            x = torch.cat(
                (cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1
            )

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
                # 推理期間，返回兩個分類器預測的平均值
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x


# ViT 權重初始化
def _init_vit_weights(m):
    # parameter m = module
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def vit_base_patch16_224(num_classes: int = 1000, num_gaussians: int = 5):
    # ViT-Base model (ViT-B/16) ImageNet-1k weights @ 224x224
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        representation_size=None,
        num_classes=num_classes,
        num_gaussians=num_gaussians,  # 傳遞 num_gaussians
    )

    return model


def vit_base_patch16_224_in21k(
    num_classes: int = 21843, has_logits: bool = True, num_gaussians: int = 5
):
    # ViT-Base model (ViT-B/16) ImageNet-21k weights @ 224x224
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        representation_size=768 if has_logits else None,
        num_classes=num_classes,
        num_gaussians=num_gaussians,  # 傳遞 num_gaussians
    )

    return model


def vit_base_patch32_224(num_classes: int = 1000, num_gaussians: int = 5):
    # ViT-Base model (ViT-B/32) ImageNet-1k weights @ 224x224
    model = VisionTransformer(
        img_size=224,
        patch_size=32,
        embed_dim=768,
        depth=12,
        num_heads=12,
        representation_size=None,
        num_classes=num_classes,
        num_gaussians=num_gaussians,  # 傳遞 num_gaussians
    )

    return model


def vit_base_patch32_224_in21k(
    num_classes: int = 21843, has_logits: bool = True, num_gaussians: int = 5
):
    # ViT-Base model (ViT-B/32) ImageNet-21k weights @ 224x224
    model = VisionTransformer(
        img_size=224,
        patch_size=32,
        embed_dim=768,
        depth=12,
        num_heads=12,
        representation_size=768 if has_logits else None,
        num_classes=num_classes,
        num_gaussians=num_gaussians,  # 傳遞 num_gaussians
    )

    return model


def vit_large_patch16_224(num_classes: int = 1000, num_gaussians: int = 5):
    # ViT-Large model (ViT-L/16) ImageNet-1k weights @ 224x224
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        representation_size=None,
        num_classes=num_classes,
        num_gaussians=num_gaussians,  # 傳遞 num_gaussians
    )

    return model


def vit_large_patch16_224_in21k(
    num_classes: int = 21843, has_logits: bool = True, num_gaussians: int = 5
):
    # ViT-Large model (ViT-L/16) ImageNet-21k weights @ 224x224
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        representation_size=1024 if has_logits else None,
        num_classes=num_classes,
        num_gaussians=num_gaussians,  # 傳遞 num_gaussians
    )

    return model


def vit_large_patch32_224_in21k(
    num_classes: int = 21843, has_logits: bool = True, num_gaussians: int = 5
):
    # ViT-Large model (ViT-L/32) ImageNet-21k weights @ 224x224
    model = VisionTransformer(
        img_size=224,
        patch_size=32,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        representation_size=1024 if has_logits else None,
        num_classes=num_classes,
        num_gaussians=num_gaussians,  # 傳遞 num_gaussians
    )

    return model


def vit_huge_patch14_224_in21k(
    num_classes: int = 21843, has_logits: bool = True, num_gaussians: int = 5
):
    # ViT-Huge model (ViT-H/14) ImageNet-21k weights @ 224x224
    model = VisionTransformer(
        img_size=224,
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        representation_size=1280 if has_logits else None,
        num_classes=num_classes,
        num_gaussians=num_gaussians,  # 傳遞 num_gaussians
    )

    return model
