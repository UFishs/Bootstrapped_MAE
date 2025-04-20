# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed
from copy import deepcopy


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, 
                 bootstrap_k=1, feature_layer=8, use_ema=False, ema_decay=0.9999, ema_warmup=0,
                 use_decoder_feature=True, soft_version=False, total_epoch=200):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE pre-training specifics
        self.bootstrap_k = bootstrap_k
        self.feature_layer = feature_layer
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.target_network = None
        self.epoch = 0
        self.ema_warmup = ema_warmup
        self.use_decoder_feature = use_decoder_feature
        self.soft_version = soft_version
        self.total_epoch = total_epoch

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch

        # decode to reconstruct embedding feature, may not be used
        self.decoder_feature = nn.Linear(decoder_embed_dim, embed_dim, bias=True)

        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

        # initialize ema network
        if self.use_ema:
            print("Using EMA for target network, warmup: ", self.ema_warmup)
            self.update_target_network()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def update_target_network(self):
        del self.target_network
        self.target_network = {}
        # deepcopy encoder
        self.target_network['patch_embed'] = deepcopy(self.patch_embed).cuda()
        self.target_network['cls_token'] = deepcopy(self.cls_token).cuda()
        self.target_network['pos_embed'] = deepcopy(self.pos_embed).cuda()
        self.target_network['blocks'] = deepcopy(self.blocks).cuda()
        self.target_network['norm'] = deepcopy(self.norm).cuda()
    
    def update_ema(self):
        # update ema network
        for key in self.target_network.keys():
            target_model = self.target_network[key]
            new_model = self.__getattr__(key)

            if isinstance(target_model,nn.Module):
                for target_param, new_param in zip(target_model.parameters(),new_model.parameters()):
                    target_param.data.copy_((1.0 - self.ema_decay) * new_param.data + self.ema_decay * target_param.data)
            else:
                target_model.copy_((1.0 - self.ema_decay) * new_model.data + self.ema_decay * target_model.data)
    
    def forward_target_encoder(self, x, feature_layer=None):
        # same as forward_encoder w/o masking
        x = self.target_network['patch_embed'](x)
        x = x + self.target_network['pos_embed'][:, 1:, :]
        cls_token = self.target_network['cls_token'] + self.target_network['pos_embed'][:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for i, blk in enumerate(self.target_network['blocks']):
            x = blk(x)
            # break when reaching the feature layer
            if feature_layer is not None and i == feature_layer:
                break
        x = self.target_network['norm'](x)
        return x


    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore, output_type='pixel'):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        if output_type == 'pixel':
            x = self.decoder_pred(x)
            x = x[:, 1:, :]  # remove cls token

        elif output_type == 'feature':
            # x: [batch_size, 8*8(32*32 with 4*4 patch)+1(cls), emb_dim]: [256, 65, 192]
            # if soft version, use decoder feature to get correct dimension
            if self.use_decoder_feature or self.soft_version:
                x = self.decoder_feature(x)
            x = x[:, 1:, :]  # remove cls token
        else:
            raise ValueError(f"Unknown output type: {output_type}")

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward_feature_loss(self, target, pred, mask):
        """
        target: [N, L*mask_ratio, embed_dim]
        pred: [N, L*mask_ratio, embed_dim]
        """
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def get_soft_alpha(self):
        """
        get the soft alpha for the soft version, must enable the soft version
        alpha = epoch / total_epoch, uniformly increase from 0 to 1
        """
        if not self.soft_version:
            raise ValueError("Soft version is not used")
        return float(self.epoch) / self.total_epoch

    def visualize_decoder_feature(self, imgs, mask_ratio=0.75):
        """
        visualize the decoder feature, used for visualization
        imgs: [N, 3, H, W]
        """
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        feat_pred = self.forward_decoder(latent, ids_restore, output_type='feature')
        feat_pixel_pred = self.decoder_pred(feat_pred)
        return feat_pixel_pred


    def forward(self, imgs, mask_ratio=0.75):

        # update ema network
        if self.use_ema:
            self.update_ema()

        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        if not self.soft_version:
            if (self.use_ema and self.epoch >= self.ema_warmup) or (self.bootstrap_k > 1 and self.target_network is not None):
                # use target network, ema or bootstrap
                pred = self.forward_decoder(latent, ids_restore, output_type='feature')  # [N, L, embed_dim]
                target = self.forward_target_encoder(imgs, feature_layer=self.feature_layer)  # [N, L, embed_dim]
                target = target[:, 1:, :]  # remove cls token
                loss = self.forward_feature_loss(target, pred, mask)

            else:
                # only pixel, simple MAE
                pred = self.forward_decoder(latent, ids_restore, output_type='pixel')  # [N, L, p*p*3]
                loss = self.forward_loss(imgs, pred, mask)
        
        else:
            # soft version, use both loss
            pixel_pred = self.forward_decoder(latent, ids_restore, output_type='pixel') 
            feature_pred = self.forward_decoder(latent, ids_restore, output_type='feature')
            feature_target = self.forward_target_encoder(imgs, feature_layer=self.feature_layer)
            feature_target = feature_target[:, 1:, :]  # remove cls token
            pixel_loss = self.forward_loss(imgs, pixel_pred, mask)
            feature_loss = self.forward_feature_loss(feature_target, feature_pred, mask)
            alpha = self.get_soft_alpha()
            loss = (1 - alpha) * pixel_loss + alpha * feature_loss
            pred = pixel_pred

        return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def deit_tiny_patch4_dec96d8b(**kwargs):
    model = MaskedAutoencoderViT(
        img_size=32, patch_size=4, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        decoder_embed_dim=192, decoder_depth=8, decoder_num_heads=3,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks

deit_tiny_patch4 = deit_tiny_patch4_dec96d8b  # decoder: 96 dim, 8 blocks
