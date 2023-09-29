from torch import nn
from .blocks import PositionalEncoder, VocabularyEmbedder
from model import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder
from .multihead_attention import MultiheadedAttention
import torch
from .utils import _get_clones, _get_activation_fn
from .bm_hrl_agent import SegmentCritic, UnimodalFusion, Worker, Manager, WorkerCore, LinearCore
from scripts.device import get_device


class DetrCaption(nn.Module):

    def __init__(self, cfg, train_dataset):
        super(DetrCaption, self).__init__()
        self.name = "detr_agent"
        self.att_layers = cfg.rl_att_layers
        self.device = get_device(cfg)
        self.dim_feedforward = 2048
        self.dif_work_man_feats = True
        self.voc_size = train_dataset.trg_voc_size
        self.d_model = cfg.d_model
        self.normalize_before = True
        self.num_layers = 3
        self.pre_goal_attention = cfg.pre_goal_attention
        self.pos_enc = PositionalEncoder(cfg.d_model, cfg.dout_p)
        self.pos_enc_C = PositionalEncoder(cfg.d_model_caps, cfg.dout_p)
        self.pos_enc_concat = PositionalEncoder(cfg.d_model_caps + cfg.rl_goal_d, cfg.dout_p)
        self.pos_enc_goal = PositionalEncoder(cfg.rl_goal_d, cfg.dout_p)
        self.n_head = cfg.rl_att_heads
        self.emb_C = VocabularyEmbedder(self.voc_size, cfg.d_model_caps)
        self.emb_C.init_word_embeddings(train_dataset.train_vocab.vectors, cfg.unfreeze_word_emb)

        self.critic = SegmentCritic(cfg)
        self.critic_score_threshhold = cfg.rl_critic_score_threshhold

        encoder_layer = TransformerEncoderLayer(cfg.d_model, self.n_head, self.dim_feedforward,
                                                cfg.dout_p, "relu", normalize_before=self.normalize_before)
        encoder_norm = nn.LayerNorm(cfg.d_model) if self.normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, self.num_layers, encoder_norm, cfg,
                                          return_intermediate=self.dif_work_man_feats)
        if not self.pre_goal_attention:
            worker_decoder_layer = decoder_layer = TransformerDecoderLayer(cfg.d_model_video, self.n_head, cfg.d_model_caps, cfg.rl_goal_d,
                                                self.dim_feedforward,
                                                cfg.dout_p, "relu", normalize_before=self.normalize_before)
            worker_decoder_norm = decoder_norm = nn.LayerNorm(cfg.d_model_caps)
        else:
            worker_decoder_layer = TransformerDecoderLayer(cfg.d_model_video, self.n_head, cfg.d_model_caps + cfg.rl_goal_d, cfg.rl_goal_d,
                                                    self.dim_feedforward,
                                                    cfg.dout_p, "relu", normalize_before=self.normalize_before)

            decoder_layer = TransformerDecoderLayer(cfg.d_model_video, self.n_head,
                                                           cfg.d_model_caps, cfg.rl_goal_d,
                                                           self.dim_feedforward,
                                                           cfg.dout_p, "relu", normalize_before=self.normalize_before)
            worker_decoder_norm = nn.LayerNorm(cfg.d_model_caps + cfg.rl_goal_d)
            decoder_norm = nn.LayerNorm(cfg.d_model_caps)
        self.worker_decoder = TransformerDecoder(worker_decoder_layer, self.num_layers, worker_decoder_norm,
                                                 return_intermediate=False)
        self.manager_decoder = TransformerDecoder(decoder_layer, self.num_layers, decoder_norm,
                                                  return_intermediate=False)

        self.manager_core = nn.Identity()
        self.manager_attention_rnn = DecoderModule(cfg.d_model_caps, cfg.rl_manager_lstm, cfg.rl_goal_d, cfg,
                                                   self.n_head, rnn=False)
        self.manager = Manager(self.device, cfg.d_model_caps, cfg.rl_goal_d, cfg.dout_p, self.manager_core)

        worker_in_d = cfg.rl_goal_d + cfg.d_model_caps
        self.worker_rnn = DecoderModule(worker_in_d, cfg.rl_worker_lstm, self.voc_size, cfg, self.n_head, rnn=False)
        self.worker_core = nn.Identity()
        self.worker = Worker(voc_size=self.voc_size, d_in=cfg.d_model_caps, d_goal=cfg.rl_goal_d, dout_p=cfg.dout_p,
                             d_model=cfg.d_model, core=self.worker_core)
        self.linear = nn.Linear(cfg.d_model_caps + cfg.rl_goal_d, self.voc_size)
        self.activation = nn.LogSoftmax(dim=-1)
        self.goal_norm = nn.LayerNorm(cfg.d_model_caps)
        self.goal_dropout = nn.Dropout(cfg.dout_p)
        self.goal_attention = MultiheadedAttention(cfg.d_model_caps, cfg.rl_goal_d, cfg.rl_goal_d, self.n_head,
                                                   cfg.dout_p, cfg.d_model)
        self.goal_feature_attention = MultiheadedAttention(cfg.rl_goal_d, cfg.d_model_caps, cfg.d_model_caps, self.n_head,
                                                           cfg.dout_p, cfg.d_model)
        self.manager_modules = [self.manager_core, self.manager_attention_rnn, self.manager, self.manager_decoder]
        self.worker_modules = [self.worker_core, self.worker, self.worker_rnn, self.worker_decoder]

        self.teaching_worker = True

    def save_model(self, checkpoint_dir):
        model_file_name = checkpoint_dir + f"/{self.name}.pt"
        torch.save(self.state_dict(), model_file_name)

    def load_model(self, checkpoint_dir):
        model_file_name = checkpoint_dir + f"/{self.name}.pt"
        self.load_state_dict(torch.load(model_file_name), strict=False)

    def _set_worker_grad(self, enabled):
        for name, param in self.worker.named_parameters():
            param.requires_grad = enabled
        for name, param in self.worker_rnn.named_parameters():
            param.requires_grad = enabled

    def _set_manager_grad(self, enabled):
        for name, param in self.manager.named_parameters():
            param.requires_grad = enabled
        for name, param in self.manager_attention_rnn.named_parameters():
            param.requires_grad = enabled

    def _set_module_grads(self, modules, enable):
        for module in modules:
            for name, param in module.named_parameters():
                param.requires_grad = enable

    def teach_worker(self):
        self.warmstarting = False
        self.teaching_worker = True
        self._set_module_grads(self.worker_modules, True)
        self._set_module_grads(self.manager_modules, False)
        self.manager.exploration = False

    def teach_manager(self):
        self.warmstarting = False
        self.teaching_worker = False
        self._set_module_grads(self.worker_modules, False)
        self._set_module_grads(self.manager_modules, True)
        self.manager.exploration = True

    def set_inference_mode(self, inference):
        if inference:
            self.manager.exploration = False
            self.worker_rnn.mode = 'inference'
            self.manager_attention_rnn.mode = 'inference'
        else:
            self.worker_rnn.mode = 'train'
            self.manager_attention_rnn.mode = 'train'
            self.manager.exploration = True

    def inference(self, x, trg, mask, worker_hid, manager_hid):
        self.worker_rnn.hidden = worker_hid
        self.manager_attention_rnn.hidden = manager_hid
        result = self.forward(x, trg, mask)[0]
        return result, self.worker_rnn.hidden, self.manager_attention_rnn.hidden

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, trg, masks, mode='train'):
        x_video, _ = x
        trg = trg.clone()
        trg[trg == 3] = 1
        C = self.emb_C(trg)
        bs, l, n_features = x_video.shape  # batchsize, length, n_features
        mask = masks['V_mask']
        memory = self.encoder(x_video, mask, self.pos_enc)

        manager_memory = worker_memory = memory
        if self.dif_work_man_feats:
            worker_memory = memory[-2]
            manager_memory = memory[-1]

        worker_context = self.manager_decoder(C, manager_memory, mask, self.pos_enc, self.pos_enc_C, masks['C_mask'],
                                              None, None, None)

        # tgt = self.embed(tgt)

        segments = self.critic(C)
        segments = torch.sigmoid(segments)
        segment_labels = (segments > self.critic_score_threshhold).squeeze().int().reshape(bs, -1)
        segment_padding = (trg == 1).sum(dim=1)

        for i in range(trg.shape[0]):
            first_end_tok = len(trg[i]) - 1 - segment_padding[i]
            segment_labels[i][first_end_tok] = 1
            segment_labels[i][first_end_tok + 1:] = 0
        goals = self.manager(worker_context, segment_labels)
        # worker_context, manager_feat = self.manager_attention_rnn(manager_feat, C, self.device, masks)
        if self.pre_goal_attention:
            goal_feature_attention = self.goal_feature_attention(self.pos_enc_goal(goals),
                                       self.pos_enc_C(C),
                                       C, masks['C_mask'])
            tgt2 = self.goal_attention(self.pos_enc_C(C),
                                       self.pos_enc_goal(goals),
                                       goals, masks['C_mask'])
            C = C + self.goal_dropout(tgt2)
            C = self.goal_norm(C)
            C_features = torch.cat([C, goal_feature_attention], dim=-1)
            worker_feat = self.worker_decoder(C_features, worker_memory, mask, self.pos_enc, self.pos_enc_concat, masks['C_mask'],
                                              None, None, None)
        else:
            worker_feat = self.worker_decoder(C, worker_memory, mask, self.pos_enc, self.pos_enc_C, masks['C_mask'],
                                              goals, masks['C_mask'], self.pos_enc_goal)
        pred = self.activation(self.linear(worker_feat))
        # goal_att = self.worker(worker_feat, goals, masks['C_mask'])
        # pred, worker_feat = self.worker_rnn(worker_feat, C, self.device, masks, True, goal_att)

        return pred, worker_feat[:, :, :300], worker_context, goals, segment_labels


class DecoderModule(nn.Module):
    def __init__(self, embed_size, hidden_size, out_size, cfg, heads, rnn=True, mode='train'):
        ''' Initialize the layers of this model.'''
        super().__init__()
        self.enc_att_V = MultiheadedAttention(cfg.d_model_caps, cfg.d_model_caps,
                                              cfg.d_model_caps, heads, cfg.dout_p, cfg.d_model)
        # Keep track of hidden_size for initialization of hidden state
        self.hidden_size = hidden_size
        self.n_head = 2
        self.mode = mode
        self.att_layers = 2
        # Embedding layer that turns words into a vector of a specified size
        # The LSTM takes embedded word vectors (of a specified size) as input
        # and outputs hidden states of size hidden_dim
        self.lstm = nn.LSTM(input_size=embed_size, \
                            hidden_size=hidden_size,  # LSTM hidden units
                            num_layers=1,  # number of LSTM layer
                            bias=True,  # use bias weights b_ih and b_hh
                            batch_first=True,
                            dropout=0,
                            bidirectional=False,  # unidirectional LSTM
                            )
        self.projection = nn.Linear(in_features=embed_size, out_features=out_size)
        self.type = 'lstm' if rnn else 'linear'
        self.hidden = None
        # The linear layer that maps the hidden state output dimension
        # to the number of words we want as output, vocab_size
        self.linear = nn.Linear(hidden_size, out_size)
        self.activation = nn.LogSoftmax(dim=-1)
        # self.activation = nn.Softmax(dim=-1)

        # initialize the hidden state
        # self.hidden = self.init_hidden()

    def init_hidden(self, batch_size, device):
        """
        So, this function defines a hidden state with all zeroes
        The axes semantics are (num_layers, batch_size, hidden_dim)
        """
        return (torch.zeros((1, batch_size, self.hidden_size), device=device), \
                torch.zeros((1, batch_size, self.hidden_size), device=device))

    def forward(self, features, caption_emb, device, masks, is_worker=False, goal_attention=None):

        # Initialize the hidden state
        batch_size = features.shape[0]  # features is of shape (batch_size, embed_size)
        if self.hidden is None or self.mode == 'train':
            self.hidden = self.init_hidden(batch_size, device)
        features = self.enc_att_V(features, caption_emb, caption_emb, masks['C_mask'])
        features_context = features

        if goal_attention is not None:
            features = torch.cat([features_context, goal_attention], dim=-1)

        # embeddings new shape : (batch_size, caption length, embed_size)
        # Get the output and hidden state by passing the lstm over our word embeddings
        # the lstm takes in our embeddings and hidden state
        if self.type == "lstm":
            lstm_out, self.hidden = self.lstm(features,
                                              self.hidden)  # lstm_out shape : (batch_size, caption length, hidden_size)

            # Fully connected layer
            outputs = self.linear(lstm_out)  # outputs shape : (batch_size, caption length, vocab_size)
        else:
            outputs = self.projection(features)
        if goal_attention is not None or is_worker:
            outputs = self.activation(outputs)
        return outputs, features_context
