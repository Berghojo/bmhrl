import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def attention(Q, K, V, mask, dropout=None):
    # Q, K, V are (B, *(H), seq_len, d_model//H = d_k)
    # mask is     (B,    1,       1,               Ss)
    d_k = Q.size(-1)
    # (B, H, S, S)

    QKt = Q.matmul(K.transpose(-1, -2))
    if torch.any(torch.isnan(QKt)):
        print(QKt, 'Q')
        raise Exception
    sm_input = QKt / np.sqrt(d_k)
    if torch.any(torch.isnan(sm_input)):
        print(sm_input, 'Q')
        raise Exception
    if torch.any(torch.isinf(sm_input)):
        print(torch.max(sm_input), 'inf')
        raise Exception
    if mask is not None:
        sm_input = sm_input.masked_fill(mask == False, -float('inf'))

    softmax = F.softmax(sm_input, dim=-1)
    softmax = torch.nan_to_num(softmax, nan=1/sm_input.size(-1))
    if torch.any(torch.isnan(softmax)):
        print(torch.max(sm_input))
        print(softmax, 'Q')
        raise Exception
    out = softmax.matmul(V)
    if torch.any(torch.isnan(out)):
        print(out, 'Q')
        raise Exception
    if dropout is not None:
        out = dropout(out)

    # (B, *(H), seq_len, d_model//H = d_k)
    return out


class MultiheadedAttention(nn.Module):

    def __init__(self, d_model_Q, d_model_K, d_model_V, H, dout_p=0.0, d_model=None):
        super(MultiheadedAttention, self).__init__()
        self.d_model_Q = d_model_Q
        self.d_model_K = d_model_K
        self.d_model_V = d_model_V
        self.H = H
        self.d_model = d_model
        self.dout_p = dout_p

        if self.d_model is None:
            print(f'd_model: is None')
            self.d_model = self.d_model_Q

        self.d_k = self.d_model // H

        self.linear_Q2d = nn.Linear(self.d_model_Q, self.d_model)
        self.linear_K2d = nn.Linear(self.d_model_K, self.d_model)
        self.linear_V2d = nn.Linear(self.d_model_V, self.d_model)
        self.linear_d2Q = nn.Linear(self.d_model, self.d_model_Q)

        self.dropout = nn.Dropout(self.dout_p)

        assert self.d_model % H == 0

    def forward(self, Q, K, V, mask):
        ''' 
            Q, K, V: (B, Sq, Dq), (B, Sk, Dk), (B, Sv, Dv)
            mask: (B, 1, Sk)
            Sk = Sv, 
            Dk != self.d_k
            Also: m1 is the target modality (queries); m2 is the source modality (keys, values)
        '''
        B, Sq, d_model_Q = Q.shape
        # (B, Sm, D) <- (B, Sm, Dm)
        Q = self.linear_Q2d(Q)
        if torch.any(torch.isnan(Q)):
            print(Q, 'Q')
            raise Exception
        K = self.linear_K2d(K)
        V = self.linear_V2d(V)

        # (B, H, Sm, d_k) <- (B, Sm, D)
        Q = Q.view(B, -1, self.H, self.d_k).transpose(-3, -2)  # (-4, -3*, -2*, -1)
        if torch.any(torch.isnan(Q)):
            print(Q, 'Q')
            raise Exception
        K = K.view(B, -1, self.H, self.d_k).transpose(-3, -2)
        if torch.any(torch.isnan(K)):
            print(K, 'K')
            raise Exception
        V = V.view(B, -1, self.H, self.d_k).transpose(-3, -2)
        if torch.any(torch.isnan(V)):
            print(V, 'V')
            raise Exception
        if mask is not None:
            # the same mask for all heads -> (B, 1, 1, Sm2)
            mask = mask.unsqueeze(1)
        # (B, H, Sq, d_k) <- (B, H, Sq, d_k), (B, H, Sk, d_k), (B, H, Sv, d_k), Sk = Sv
        Q = attention(Q, K, V, mask, self.dropout)
        if torch.any(torch.isnan(Q)):
            print(Q, 'Q')
            raise Exception
        # (B, Sq, D) <- (B, H, Sq, d_k)
        Q = Q.transpose(-3, -2).contiguous().view(B, Sq, self.d_model)
        if torch.any(torch.isnan(Q)):
            print(Q, 'Q')
            raise Exception
        # (B, Sq, Dq)
        Q = self.linear_d2Q(Q)
        if torch.any(torch.isnan(Q)):
            print(Q, 'Q')
            raise Exception

        return Q