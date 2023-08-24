import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger
from mmcv.runner import load_checkpoint
import math
from ..builder import BACKBONES, build_backbone
class SEAttention(nn.Module):
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=True),
            nn.Sigmoid()
        )
        # self.fc_linear1 = nn.Linear(channel, channel // reduction, bias=True)
        # self.fc_ReLU = nn.ReLU(inplace=True)
        # self.fc_linear2 = nn.Linear(channel // reduction, channel, bias=True)
        # self.fc_Sigmoid = nn.Sigmoid()
        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # y = self.fc_linear1(y)
        # y = self.fc_ReLU(y)
        # y = self.fc_linear2(y)
        # y = self.fc_Sigmoid(y).view(b, c, 1, 1)
        return x * y

class Efficent_FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.se = SEAttention(out_features)
        self.mffn = nn.Sequential(
            # pw
            nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True),
            nn.BatchNorm2d(hidden_features),
            nn.SiLU(),
            # dw
            nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features, bias=True),
            nn.BatchNorm2d(hidden_features),
            nn.SiLU(),
            # pw-linear
            nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True),
            nn.BatchNorm2d(out_features)
        )
        # self.mffn_conv1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        # self.mffn_bn1 = nn.BatchNorm2d(hidden_features)
        # self.mffn_act1 = nn.SiLU()
        # self.mffn_conv2 = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features, bias=True)
        # self.mffn_bn2 = nn.BatchNorm2d(hidden_features)
        # self.mffn_act2 = nn.SiLU()
        # self.mffn_conv3 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        # self.mffn_bn3 = nn.BatchNorm2d(out_features)

        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.mffn(x)
        # x = self.mffn_conv1(x)
        # x = self.mffn_bn1(x)
        # x = self.mffn_act1(x)
        # x = self.mffn_conv2(x)
        # x = self.mffn_bn2(x)
        # x = self.mffn_act2(x)
        # x = self.mffn_conv3(x)
        # x = self.mffn_bn3(x)

        x = self.se(x)
        x = self.drop(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Dual_Granularity_Former_Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Dual_Granularity_Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Efficent_FFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        assert max(patch_size) > stride, "Set larger patch_size than stride"

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W

class AbsolutePositionEmbedding(nn.Module):
    def __init__(self, pos_shape, pos_dim, drop_rate=0.):
        super().__init__()

        pos_shape = to_2tuple(pos_shape)
        self.pos_shape = pos_shape
        self.pos_dim = pos_dim

        self.pos_embed = nn.Parameter(
            torch.zeros(1, pos_shape[0] * pos_shape[1], pos_dim))
        self.drop = nn.Dropout(p=drop_rate)

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)

    def resize_pos_embed(self, pos_embed, H, W, mode='bilinear'):
        """Resize pos_embed weights.

        Resize pos_embed using bilinear interpolate method.

        Args:
            pos_embed (torch.Tensor): Position embedding weights.
            input_shape (tuple): Tuple for (downsampled input image height,
                downsampled input image width).
            mode (str): Algorithm used for upsampling:
                ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                ``'trilinear'``. Default: ``'bilinear'``.

        Return:
            torch.Tensor: The resized pos_embed of shape [B, L_new, C].
        """
        assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
        pos_h, pos_w = self.pos_shape
        pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w):]
        pos_embed_weight = pos_embed_weight.reshape(1, pos_h, pos_w, self.pos_dim).permute(0, 3, 1, 2).contiguous()
        pos_embed_weight = F.interpolate(pos_embed_weight, size=(H, W), mode=mode)
        pos_embed_weight = torch.flatten(pos_embed_weight,2).transpose(1, 2).contiguous()
        pos_embed = pos_embed_weight

        return pos_embed

    def forward(self, x, hw_shape, mode='bilinear'):
        pos_embed = self.resize_pos_embed(self.pos_embed, hw_shape, mode)
        return self.drop(x + pos_embed)

class ConvStem(nn.Module):
    def __init__(self, in_channels, embed_dims_i):
        super(ConvStem, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, embed_dims_i, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(embed_dims_i)
        self.act1 = nn.SiLU()
        self.conv_dw1 = nn.Conv2d(embed_dims_i, embed_dims_i, 3, 2, 1, groups=embed_dims_i, bias=False)
        self.bn2 = nn.BatchNorm2d(embed_dims_i)
        self.act2 = nn.SiLU()
        self.conv_pw1 = nn.Conv2d(embed_dims_i, embed_dims_i // 2, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(embed_dims_i // 2)
        self.conv_pw2 = nn.Conv2d(embed_dims_i // 2, embed_dims_i * 2, 1, 1, 0, bias=False)
        self.bn4 = nn.BatchNorm2d(embed_dims_i * 2)
        self.act3 = nn.SiLU()
        self.conv_dw2 = nn.Conv2d(embed_dims_i * 2, embed_dims_i * 2, 3, 2, 1, groups=embed_dims_i * 2, bias=False)
        self.bn5 = nn.BatchNorm2d(embed_dims_i * 2)
        self.act4 = nn.SiLU()
        self.conv_pw3 = nn.Conv2d(embed_dims_i * 2, embed_dims_i, 1, 1, 0, bias=False)
        self.bn6 = nn.BatchNorm2d(embed_dims_i)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv_dw1(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.conv_pw1(x)
        x = self.bn3(x)
        x = self.conv_pw2(x)
        x = self.bn4(x)
        x = self.act3(x)
        x = self.conv_dw2(x)
        x = self.bn5(x)
        x = self.act4(x)
        x = self.conv_pw3(x)
        x = self.bn6(x)
        return x

# PEG  from https://arxiv.org/abs/2102.10882
class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim), )
        self.s = s

    def forward(self, x, H, W):
        B, N, C = x.shape
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]

class Dual_Granularity_Former(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000, embed_dims=[32, 64, 160, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1], num_stages=4, linear=False, pretrained=None):
        super().__init__()
        # self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.linear = linear
        self.pos_block = nn.ModuleList(
            [PosCNN(embed_dim, embed_dim) for embed_dim in [32, 64, 160, 256]]
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = nn.Sequential(
                    # nn.Conv2d(in_channels=in_channels, out_channels=embed_dims_i // 2, kernel_size=3, stride=2,
                    #           padding=1),
                    # act_cfg(),
                    # nn.Conv2d(in_channels=embed_dims_i // 2, out_channels=embed_dims_i // 2, kernel_size=(3, 3),
                    #           stride=(1, 1),
                    #           padding=(1, 1)),
                    # act_cfg(),
                    # nn.Conv2d(in_channels=embed_dims_i // 2, out_channels=embed_dims_i, kernel_size=(3, 3),
                    #           stride=(2, 2),
                    #           padding=(1, 1)),
                    # act_cfg(),
                    # nn.Conv2d(in_channels=embed_dims_i, out_channels=embed_dims_i, kernel_size=(3, 3),
                    #           stride=(1, 1), padding=(1, 1)),
                    # act_cfg(),
                    nn.Conv2d(in_channels, embed_dims[0], 1, 1, 0, bias=False),
                    nn.BatchNorm2d(embed_dims[0]),
                    nn.SiLU(),
                    # dw
                    nn.Conv2d(embed_dims[0], embed_dims[0], 3, 2, 1, groups=embed_dims[0], bias=False),
                    nn.BatchNorm2d(embed_dims[0]),
                    nn.SiLU(),
                    # pw-linear
                    nn.Conv2d(embed_dims[0], embed_dims[0] // 2, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(embed_dims[0] // 2),

                    # pw
                    nn.Conv2d(embed_dims[0] // 2, embed_dims[0] * 2, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(embed_dims[0] * 2),
                    nn.SiLU(),
                    # dw
                    nn.Conv2d(embed_dims[0] * 2, embed_dims[0] * 2, 3, 2, 1, groups=embed_dims[0] * 2, bias=False),
                    nn.BatchNorm2d(embed_dims[0] * 2),
                    nn.SiLU(),
                    # pw-linear
                    nn.Conv2d(embed_dims[0] * 2, embed_dims[0], 1, 1, 0, bias=False),
                    nn.BatchNorm2d(embed_dims[0]),
                )
                # patch_embed = ConvStem(in_channels, embed_dims[0])
            else:
                patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                                patch_size=7 if i == 0 else 3,
                                                stride=4 if i == 0 else 2,
                                                in_chans=embed_dims[i - 1],
                                                embed_dim=embed_dims[i])
            #####
            # block = nn.ModuleList()
            # patch_sizes = [7, 3, 3, 3]
            # pos_shape = img_size // np.prod(patch_sizes[:i + 1])
            # pos_embed = AbsolutePositionEmbedding(
            #     pos_shape=pos_shape,
            #     pos_dim=embed_dims[i],
            #     drop_rate=drop_rate)
            # block.append(pos_embed)
            # block.extend([Dual_Granularity_Former_Block(
            #     dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
            #     qk_scale=qk_scale,
            #     drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
            #     sr_ratio=sr_ratios[i])
            #     for j in range(depths[i])])
            pos_block = nn.ModuleList()
            patch_sizes = [4,2,2,2]
            pos_shape = img_size // np.prod(patch_sizes[:i + 1])
            pos_embed = AbsolutePositionEmbedding(
                pos_shape=pos_shape,
                pos_dim=embed_dims[i],
                drop_rate=drop_rate)
            pos_block.append(pos_embed)
            ###


            block = nn.ModuleList([Dual_Granularity_Former_Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i])
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"pos_block{i + 1}", pos_block)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        # self.head = nn.Linear(embed_dims * num_heads[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        self.init_weights(pretrained)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)


    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        # return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better
        return set(['cls_token'] + ['pos_block.' + n for n, p in self.pos_block.named_parameters()])

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            pos_block = getattr(self, f"pos_block{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            if i == 0:
                x = patch_embed(x)
                _, _, H, W = x.shape
                x = x.flatten(2).transpose(1, 2)
            else:
                x, H, W = patch_embed(x)
            for pos in pos_block:
                x = pos(x, H, W)
            # for blk in block:
            for j, blk in enumerate(block):
                x = blk(x, H, W)
                if j == 0:
                    x = self.pos_block[i](x, H, W)  # PEG here
            x = norm(x)
            # if i != self.num_stages - 1:
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)

        return x

class Dual_Granularity_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, num_focal_levels=2, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        self.num_focal_levels = num_focal_levels
        if sr_ratio > 1:
            self.sr = nn.ModuleList([
                # nn.Conv2d(dim, dim, kernel_size=sr_ratio // (i+1), stride=sr_ratio // (i+1))
                nn.MaxPool2d(kernel_size=sr_ratio // (i+1), stride=sr_ratio // (i+1))
                for i in range(num_focal_levels)
            ])
            self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_focal_levels)])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape # 1,784,64
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # 1,4, 784, 16

        if self.sr_ratio > 1:  # 6
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W) # 1, 64, 28, 28
            kv_list = []
            for i in range(self.num_focal_levels):
                x_sr = self.sr[i](x_).reshape(B, C, -1).permute(0, 2, 1) # 1, 81, 64 + 1, 16, 64
                x_sr = self.norms[i](x_sr)
                kv_list.append(self.kv(x_sr).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4))# 2,1,4,748,16
            kv = torch.cat(kv_list, dim=-2) # 2,1,4,81,16 + 2,1,4,16,16
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # 1, 97, 64
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict




@BACKBONES.register_module()
class dg_former(Dual_Granularity_Former):
    def __init__(self, **kwargs):
        super(dg_former, self).__init__(
            patch_size=4, embed_dims=16, num_heads=[2, 4, 10, 16], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 6, 4, 2],
            drop_rate=0.0, drop_path_rate=0.1, pretrained=kwargs['pretrained'])