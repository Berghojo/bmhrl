from .multihead_attention import MultiheadedAttention
from torch.nn import MultiheadAttention as MHA
import torch.nn.functional as F
import copy
from torch import nn
import torch
from .utils import _get_clones,_get_activation_fn


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, cfg=None):
        super().__init__()
        self.cfg = cfg
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask, pos_enc):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask, pos=pos_enc)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=True, embed_size=300):
        super().__init__()
        self.self_attn = MultiheadedAttention(d_model, d_model, d_model, nhead, dropout, d_model)
        #self.self_attn = MHA(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.embed = nn.Linear(d_model, embed_size)
        self.norm1 = nn.LayerNorm(1024)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     mask, pos):
        q = k = pos(src)
        src2 = self.self_attn(q, k, src, mask)
        src = src + self.dropout1(src2)
        # src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward(self, src,
                src_mask,
                pos):

        return self.forward_post(src, src_mask, pos)

