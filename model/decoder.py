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

    def forward(self, tgt, memory, mask, pos, query_pos, query_mask, goal, goal_mask, goal_pos=None):
        output = tgt
        intermediate = []
        for layer in self.layers:
            output = layer(output, memory, mask, pos, query_pos, query_mask, goal, goal_mask, goal_pos)

            if self.return_intermediate:

                intermediate.append(output)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(self.norm(output))

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, d_model_C, d_goal, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = MultiheadedAttention(d_model_C, d_model_C, d_model_C, nhead, dropout, d_model)
        self.multihead_attn = MultiheadedAttention(d_model_C, d_model, d_model, nhead, dropout, d_model)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model_C, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model_C)
        self.norm1 = nn.LayerNorm(d_model_C)
        self.norm2 = nn.LayerNorm(d_model_C)
        self.norm3 = nn.LayerNorm(d_model_C)
        self.norm4 = nn.LayerNorm(d_model_C)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.goal_attention = MultiheadedAttention(d_model_C, d_goal, d_goal, nhead, dropout, d_model)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def forward_post(self, tgt, memory,
                     memory_mask,
                     pos,
                     query_pos , query_mask, goal, goal_mask, goal_pos):

        q = k = query_pos(tgt)
        tgt2 = self.self_attn(q, k, tgt, query_mask)
        tgt = self.norm1(tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.multihead_attn(query_pos(tgt),
                                   pos(memory),
                                   memory, memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        if goal is not None:
            tgt2 = self.goal_attention(query_pos(tgt),
                                       goal_pos(goal),
                                       goal, None)
            tgt = tgt + self.dropout4(tgt2)
            tgt = self.norm4(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, memory,
                memory_mask,
                pos,
                query_pos, query_mask, goal, goal_mask, goal_pos):
        return self.forward_post(tgt, memory, memory_mask, pos, query_pos, query_mask, goal, goal_mask, goal_pos)

