from torch import nn
from model import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder
from torchvision.models import VisionTransformer
import torch
from .blocks import PositionalEncoder


class ObjectDetect(nn.Module):
    def __init__(self, cfg, voc_size):
        super().__init__()
        self.d_model = cfg.d_model
        hidden_dim = 256
        num_queries = 100
        self.num_classes = voc_size + 1
        self.class_embed = nn.Linear(hidden_dim, self.num_classes)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.pos_enc = PositionalEncoder(hidden_dim, cfg.dout_p)
        self.input_projection = nn.Linear(self.d_model, hidden_dim)
        encoder_layer = TransformerEncoderLayer(hidden_dim, 4, 2048,
                                                cfg.dout_p, "relu", normalize_before=True)
        encoder_norm = nn.LayerNorm(hidden_dim)
        self.encoder = TransformerEncoder(encoder_layer, 6, encoder_norm, cfg,
                                          return_intermediate=False)

        decoder_layer = TransformerDecoderLayer(hidden_dim, 4,
                                                hidden_dim, cfg.rl_goal_d,
                                                2048,
                                                cfg.dout_p, "relu", normalize_before=True)
        decoder_norm = nn.LayerNorm(hidden_dim)
        self.linear = nn.Linear(hidden_dim, voc_size)

        self.decoder = TransformerDecoder(decoder_layer, 6, decoder_norm,
                                                 return_intermediate=False)

    def forward(self, samples, mask):
        samples = self.input_projection(samples)
        bs, sl, f = samples.shape
        memory = self.encoder(samples, mask, self.pos_enc)
        query_pos = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        tgt = torch.zeros_like(query_pos)
        hs = self.decoder(tgt, memory, mask, self.pos_enc, query_pos, None,
                                              None, None, None, add_pos=True)
        predicted_words = self.class_embed(hs)
        attention_mask = (torch.argmax(predicted_words.softmax(-1), -1) == (self.num_classes - 1))

        return predicted_words, hs.detach(), attention_mask.detach()
