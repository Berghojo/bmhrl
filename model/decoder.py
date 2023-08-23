from torch import nn
from .multihead_attention import MultiheadedAttention
from .utils import _get_clones, _get_activation_fn
import torch


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=True):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory, mask, pos, query_pos):
        output = tgt
        intermediate = []
        for layer in self.layers:
            output = layer(output, memory, mask, pos, query_pos)
            if torch.any(torch.isnan(output)):
                print(output, 'res')
                output = torch.nan_to_num(output)
                #TODO find reason for random nans in output

                #raise Exception
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = MultiheadedAttention(d_model, d_model, d_model, nhead, dropout, d_model)
        self.multihead_attn = MultiheadedAttention(d_model, d_model, d_model, nhead, dropout, d_model)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask,
                     pos,
                     query_pos):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1), None)
        tgt = self.norm1(tgt).transpose(0, 1)
        tgt = tgt + self.dropout1(tgt)
        tgt2 = self.multihead_attn(self.with_pos_embed(tgt, query_pos),
                                   self.with_pos_embed(memory, pos),
                                   memory, memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, memory,
                memory_mask,
                pos,
                query_pos):
        return self.forward_post(tgt, memory, memory_mask, pos, query_pos)

