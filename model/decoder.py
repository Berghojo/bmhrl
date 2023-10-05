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

    def forward(self, tgt, memory, mask, pos, query_pos, query_mask, goal, goal_mask, goal_pos=None, add_pos=False,
                detected_objects=None, obj_mask=None,):
        output = tgt
        intermediate = []
        for layer in self.layers:
            output = layer(output, memory, mask, pos, query_pos, query_mask, goal, goal_mask, goal_pos, add_pos,
                           detected_objects, obj_mask)

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
        self.detected_attention = MultiheadedAttention(d_model_C, 256, 256, nhead, dropout, d_model)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model_C, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model_C)
        self.norm1 = nn.LayerNorm(d_model_C)
        self.norm2 = nn.LayerNorm(d_model_C)
        self.norm3 = nn.LayerNorm(d_model_C)
        self.norm4 = nn.LayerNorm(d_model_C)
        self.norm5 = nn.LayerNorm(d_model_C)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.dropout5 = nn.Dropout(dropout)
        self.goal_attention = MultiheadedAttention(d_model_C, d_goal, d_goal, nhead, dropout, d_model)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.positional_encoding = nn.Parameter()

    def forward_post(self, tgt, memory,
                     memory_mask,
                     pos,
                     query_pos, query_mask, goal, goal_mask, goal_pos, detected_objects=None, add_pos=False, obj_mask=None):
        if not add_pos:
            causal = True
            q = k = query_pos(tgt)
        else:
            q = k = tgt + query_pos
            causal = False
        tgt2 = self.self_attn(q, k, tgt, query_mask, causal=causal)
        tgt = self.norm1(tgt)
        tgt = tgt + self.dropout1(tgt2)

        tgt2 = self.multihead_attn(q,
                                   pos(memory),
                                   memory, memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        if goal is not None:
            tgt2 = self.goal_attention(query_pos(tgt),
                                       goal_pos(goal),
                                       goal, goal_mask)
            tgt = tgt + self.dropout4(tgt2)
            tgt = self.norm4(tgt)
        if detected_objects is not None:
            tgt2 = self.detected_attention(q,
                                           detected_objects,
                                           detected_objects, obj_mask)
            tgt = tgt + self.dropout5(tgt2)
            tgt = self.norm5(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, memory,
                memory_mask,
                pos,
                query_pos, query_mask, goal, goal_mask, goal_pos, add_pos=False, detected_objects=None, obj_mask=None):
        return self.forward_post(tgt, memory, memory_mask, pos, query_pos, query_mask, goal, goal_mask, goal_pos,
                                 add_pos=add_pos, detected_objects=detected_objects, obj_mask=None)
