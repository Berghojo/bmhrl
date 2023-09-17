import torch
import torch.nn as nn
import torch.nn.functional as F

# Provided by Iashin and Rahtu at https://github.com/v-iashin/BMT
from model.blocks import LayerStack, PositionalEncoder, PositionwiseFeedForward, ResidualConnection, VocabularyEmbedder, \
    clone
from model.multihead_attention import MultiheadedAttention
# ---------------------------------------------------------------
from scripts.device import get_device


class AReLU(nn.Module):
    def __init__(self, alpha=0.90, beta=2.0):
        super(AReLU, self).__init__()
        self.alpha = nn.Parameter(torch.tensor([alpha]))
        self.beta = nn.Parameter(torch.tensor([beta]))

    def forward(self, input):
        alpha = torch.clamp(self.alpha, min=0.01, max=0.99)
        beta = 1 + torch.sigmoid(self.beta)

        return F.relu(input) * beta - F.relu(-input) * alpha


class ModelBase(nn.Module):
    def __init__(self, name) -> None:
        super(ModelBase, self).__init__()
        self.name = name

    def save_model(self, checkpoint_dir):
        model_file_name = checkpoint_dir + f"/{self.name}.pt"
        torch.save(self.state_dict(), model_file_name)

    def load_model(self, checkpoint_dir):
        model_file_name = checkpoint_dir + f"/{self.name}.pt"
        self.load_state_dict(torch.load(model_file_name))


class ModalityProjection(nn.Module):
    def __init__(self, d_mod, d_out, p_dout) -> None:
        super(ModalityProjection, self).__init__()
        self.linear = nn.Linear(d_mod, d_out)
        self.norm = nn.LayerNorm(d_mod)
        self.dropout = nn.Dropout(p_dout)

    def forward(self, x):
        x = self.norm(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class BMFusionLayer(nn.Module):
    def __init__(self, d_model_A, d_model_V, d_model_C, d_model, d_ff_c, dout_p, H) -> None:
        super(BMFusionLayer, self).__init__()
        # encoder attention
        self.res_layer_self_att = ResidualConnection(d_model_C, dout_p)
        self.self_att = MultiheadedAttention(d_model_C, d_model_C, d_model_C, H, dout_p, d_model)

        self.res_layer_enc_att_A = ResidualConnection(d_model_C, dout_p)
        self.res_layer_enc_att_V = ResidualConnection(d_model_C, dout_p)
        self.enc_att_A = MultiheadedAttention(d_model_C, d_model_A, d_model_A, H, dout_p, d_model)
        self.enc_att_V = MultiheadedAttention(d_model_C, d_model_V, d_model_V, H, dout_p, d_model)

        self.feed_forward = PositionwiseFeedForward(d_model_C, d_ff_c, dout_p)

        self.normCA = nn.LayerNorm(d_model_C)
        self.normCV = nn.LayerNorm(d_model_C)

        self.a_v_constant = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x, masks):
        '''
        Inputs:
            x (C, memory): C: (B, Sc, Dc)
                           memory: (Av: (B, Sa, Da), Va: (B, Sv, Dv))
            masks (V_mask: (B, 1, Sv); A_mask: (B, 1, Sa); C_mask (B, Sc, Sc))
        Outputs:
            x (C, memory): C: (B, Sc, Dc)
                           memory: (Av: (B, Sa, Da), Va: (B, Sv, Dv))
        '''
        C, memory = x

        Av, Va = memory

        # Define sublayers
        # a comment regarding the motivation of the lambda function please see the EncoderLayer
        def sublayer_self_att(C): return self.self_att(C, C, C, masks['C_mask'])

        def sublayer_enc_att_A(C):
            return self.enc_att_A(C, Av, Av, masks['A_mask'])

        def sublayer_enc_att_V(C): return self.enc_att_V(C, Va, Va, masks['V_mask'])
        # sublayer_feed_forward = self.feed_forward

        # 1. Self Attention
        # (B, Sc, Dc)
        C = self.res_layer_self_att(C, sublayer_self_att)

        # 2. Encoder-Decoder Attention
        # (B, Sc, Dc) each
        Ca = self.res_layer_enc_att_A(C, sublayer_enc_att_A)
        Cv = self.res_layer_enc_att_V(C, sublayer_enc_att_V)

        # Norm individually
        Ca = self.normCA(Ca)
        Cv = self.normCV(Cv)

        # TODO check how this performance, possibly go bakc to cocnatenation or fix .5/.5 ratio
        av_factor = torch.sigmoid(torch.clamp(self.a_v_constant, min=-2, max=2))

        # B seq_len 300
        fused_features = av_factor * Cv + (1 - av_factor) * Ca

        # TODO Mby add FF Layer
        return fused_features, memory


class BMFusion(nn.Module):
    def __init__(self, d_model_A, d_model_V, d_model_C, d_model, d_ff_c, dout_p, H, N) -> None:
        super(BMFusion, self).__init__()
        layer = BMFusionLayer(
            d_model_A, d_model_V, d_model_C, d_model, d_ff_c, dout_p, H
        )
        self.decoder = LayerStack(layer, N)

    def forward(self, x, masks):
        C, memory = self.decoder(x, masks)
        return C


class UnimodalFusion(nn.Module):
    def __init__(self, d_model_M1, d_model_C, d_model, d_ff_c, dout_p, H, N) -> None:
        super(UnimodalFusion, self).__init__()
        layer = UnimodalFusionLayer(
            d_model_M1, d_model_C, d_model, d_ff_c, dout_p, H
        )
        self.decoder = LayerStack(layer, N)

    def forward(self, x, masks):
        C, memory = self.decoder(x, masks)

        return C


class UnimodalFusionLayer(nn.Module):
    def __init__(self, d_model_M1, d_model_C, d_model, d_ff_c, dout_p, H) -> None:
        super(UnimodalFusionLayer, self).__init__()
        self.res_layer_self_att = ResidualConnection(d_model_C, dout_p)
        self.self_att = MultiheadedAttention(d_model_C, d_model_C, d_model_C, H, dout_p, d_model)

        self.res_layer_enc_att = ResidualConnection(d_model_C, dout_p)

        self.enc_att = MultiheadedAttention(d_model_C, d_model_M1, d_model_M1, H, dout_p, d_model)

        self.feed_forward = PositionwiseFeedForward(d_model_C, d_ff_c, dout_p)

        self.normC = nn.LayerNorm(d_model_C)

    def forward(self, x, masks):
        '''
        Inputs:
            x (C, memory): C: (B, Sc, Dc)
                           memory: (Av: (B, Sa, Da), Va: (B, Sv, Dv))
            masks (V_mask: (B, 1, Sv); A_mask: (B, 1, Sa); C_mask (B, Sc, Sc))
        Outputs:
            x (C, memory): C: (B, Sc, Dc)
                           memory: (Av: (B, Sa, Da), Va: (B, Sv, Dv))
        '''
        C, memory = x
        M1 = memory

        m1_mask, c_mask = masks

        def sublayer_self_att(C): return self.self_att(C, C, C, c_mask)

        def sublayer_enc_att(C): return self.enc_att(C, M1, M1, m1_mask)

        C = self.res_layer_self_att(C, sublayer_self_att)
        Cm = self.res_layer_enc_att(C, sublayer_enc_att)
        Cm = self.normC(Cm)
        return Cm, memory


class SegmentCritic(nn.Module):
    def __init__(self, cfg):
        super(SegmentCritic, self).__init__()
        self.name = "SegmentCritic"
        embed_dim = cfg.d_model_caps

        self.lstm = nn.LSTM(embed_dim, 2 * embed_dim, num_layers=4, batch_first=True)
        self.gru = nn.GRU(2 * embed_dim, 2 * embed_dim, num_layers=2, batch_first=True)
        self.lin = nn.Linear(2 * embed_dim, 1)
        self.relu = AReLU()
        self.relu2 = AReLU()

        for name, param in self.named_parameters():
            param.requires_grad = False

        self.load_state_dict(torch.load(cfg.rl_critic_path))
        print('loaded: ', cfg.rl_critic_path)

    def forward(self, embedded_indices):
        # Pretrained Critic
        with torch.no_grad():
            h_1, _ = self.lstm(embedded_indices)

            h_1 = self.relu(h_1)
            h_2, _ = self.gru(h_1)

            h_2 = self.relu2(h_2)
            x = self.lin(h_2)

            return x


class BMEncoder(nn.Module):
    def __init__(self, d_model_M1, d_model_M2, d_model, d_ff_M1, d_ff_M2, dout_p, H, N) -> None:
        super(BMEncoder, self).__init__()
        enc_layer = BMEncoderLayer(d_model_M1, d_model_M2, d_model, d_ff_M1, d_ff_M2, dout_p, H)
        self.encoder = LayerStack(enc_layer, N)

    def forward(self, x, masks):
        '''
        in:
            x: (B, S, d_model) src_mask: (B, 1, S)
        out:
            # x: (B, S, d_model) which will be used as Q and K in Fusion step
        '''
        V, A = x

        # M1m2 (B, Sm1, D), M2m1 (B, Sm2, D) <-
        Av, Va = self.encoder((V, A), (masks['V_mask'], masks['A_mask']))
        return (Av, Va)


class UnimodalEncoder(nn.Module):
    def __init__(self, d_model_M1, d_model, d_ff_M1, dout_p, H, N) -> None:
        super(UnimodalEncoder, self).__init__()
        enc_layer = UnimodalEncoderLayer(d_model_M1, d_model, d_ff_M1, dout_p, H)
        self.encoder = LayerStack(enc_layer, N)

    def forward(self, x, mask):
        # M1m2 (B, Sm1, D), M2m1 (B, Sm2, D) <-
        x = self.encoder(x, mask)

        return x


class BMWorkerValueFunction(ModelBase):
    def __init__(self, cfg) -> None:
        super(BMWorkerValueFunction, self).__init__("bm_worker_value_function")
        d_goal = cfg.rl_goal_d
        d_worker_feat = cfg.d_model_caps
        dout_p = cfg.dout_p

        input_d = d_worker_feat + d_goal
        self.value_function = PositionwiseFeedForward(input_d, input_d * 2, dout_p)
        self.projection = nn.Linear(input_d, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        w_feat, goal = x
        predicted_value = self.value_function(torch.cat([w_feat, goal], dim=-1))
        predicted_value = self.activation(predicted_value)
        predicted_value = self.projection(predicted_value)
        return predicted_value  # self.scaler(predicted_value)


class BMManagerValueFunction(ModelBase):
    def __init__(self, cfg) -> None:
        super(BMManagerValueFunction, self).__init__("bm_manager_value_function")
        d_manager_feat = cfg.d_model_caps
        dout_p = cfg.dout_p

        self.value_function = PositionwiseFeedForward(d_manager_feat, d_manager_feat * 2, dout_p)
        self.projection = nn.Linear(d_manager_feat, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        predicted_value = self.value_function(x)
        predicted_value = self.activation(predicted_value)
        predicted_value = self.projection(predicted_value)
        return predicted_value  # self.scaler(predicted_value)


class UnimodalEncoderLayer(nn.Module):

    def __init__(self, d_M1, d_model, d_ff_M1, dout_p, H):
        super(UnimodalEncoderLayer, self).__init__()
        self.self_att_M1 = MultiheadedAttention(d_M1, d_M1, d_M1, H, dout_p, d_model)

        # With Nonlinearity
        self.feed_forward_M1 = PositionwiseFeedForward(d_M1, d_ff_M1, dout_p)

        self.res_layers_M1 = clone(ResidualConnection(d_M1, dout_p), 3)

    def forward(self, x, masks):
        '''
        Inputs:
            x (M1, M2): (B, Sm, Dm)
            masks (M1, M2): (B, 1, Sm)
        Output:
            M1m2 (B, Sm1, Dm1), M2m1 (B, Sm2, Dm2),
        '''
        M1 = x
        M1_mask = masks

        # sublayer should be a function which inputs x and outputs transformation
        # thus, lambda is used instead of just `self.self_att(x, x, x)` which outputs
        # the output of the self attention
        def sublayer_self_att_M1(M1): return self.self_att_M1(M1, M1, M1, M1_mask)

        sublayer_ff_M1 = self.feed_forward_M1

        # 1. Self-Attention
        # bot
        # h (B, Sm*, Dm*)
        M1 = self.res_layers_M1[0](M1, sublayer_self_att_M1)

        M1 = self.res_layers_M1[2](M1, sublayer_ff_M1)

        return M1


class BMEncoderLayer(nn.Module):

    def __init__(self, d_model_M1, d_model_M2, d_model, d_ff_M1, d_ff_M2, dout_p, H):
        super(BMEncoderLayer, self).__init__()
        self.self_att_M1 = MultiheadedAttention(d_model_M1, d_model_M1, d_model_M1, H, dout_p, d_model)
        self.self_att_M2 = MultiheadedAttention(d_model_M2, d_model_M2, d_model_M2, H, dout_p, d_model)
        self.bi_modal_att_M1 = MultiheadedAttention(d_model_M1, d_model_M2, d_model_M2, H, dout_p, d_model)
        self.bi_modal_att_M2 = MultiheadedAttention(d_model_M2, d_model_M1, d_model_M1, H, dout_p, d_model)

        # With Nonlinearity
        self.feed_forward_M1 = PositionwiseFeedForward(d_model_M1, d_ff_M1, dout_p)
        self.feed_forward_M2 = PositionwiseFeedForward(d_model_M2, d_ff_M2, dout_p)

        self.res_layers_M1 = clone(ResidualConnection(d_model_M1, dout_p), 3)
        self.res_layers_M2 = clone(ResidualConnection(d_model_M2, dout_p), 3)

    def forward(self, x, masks):
        '''
        Inputs:
            x (M1, M2): (B, Sm, Dm)
            masks (M1, M2): (B, 1, Sm)
        Output:
            M1m2 (B, Sm1, Dm1), M2m1 (B, Sm2, Dm2),
        '''
        M1, M2 = x
        M1_mask, M2_mask = masks

        # sublayer should be a function which inputs x and outputs transformation
        # thus, lambda is used instead of just `self.self_att(x, x, x)` which outputs
        # the output of the self attention
        def sublayer_self_att_M1(M1): return self.self_att_M1(M1, M1, M1, M1_mask)

        def sublayer_self_att_M2(M2): return self.self_att_M2(M2, M2, M2, M2_mask)

        def sublayer_att_M1(M1): return self.bi_modal_att_M1(M1, M2, M2, M2_mask)

        def sublayer_att_M2(M2): return self.bi_modal_att_M2(M2, M1, M1, M1_mask)

        sublayer_ff_M1 = self.feed_forward_M1
        sublayer_ff_M2 = self.feed_forward_M2

        # 1. Self-Attention
        # both (B, Sm*, Dm*)
        M1 = self.res_layers_M1[0](M1, sublayer_self_att_M1)
        M2 = self.res_layers_M2[0](M2, sublayer_self_att_M2)
        # 2. Multimodal Attention (var names: M* is the target modality; m* is the source modality)
        # (B, Sm1, Dm1)
        M1m2 = self.res_layers_M1[1](M1, sublayer_att_M1)
        # (B, Sm2, Dm2)
        M2m1 = self.res_layers_M2[1](M2, sublayer_att_M2)

        # (B, Sm1, Dm1)
        M1m2 = self.res_layers_M1[2](M1m2, sublayer_ff_M1)
        # (B, Sm2, Dm2)
        M2m1 = self.res_layers_M2[2](M2m1, sublayer_ff_M2)

        return M1m2, M2m1


class LinearCore(nn.Module):
    def __init__(self, d_model_caps, d_goal, dout_p) -> None:
        super(LinearCore, self).__init__()
        self.linear = nn.Linear(d_model_caps, d_goal)
        self.dropout = nn.Dropout(dout_p)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        return x


class Manager(nn.Module):
    def __init__(self, device, d_model_caps, d_goal, dout_p, core = None, exploration=True, ) -> None:
        super(Manager, self).__init__()
        self.device = device
        if core is not None:
            self.core = core
        else:
            self.core = LinearCore(d_model_caps, d_goal, dout_p)
        self.linear = nn.Linear(d_model_caps, d_goal)
        self.dropout = nn.Dropout(dout_p)
        self.exploration = exploration
        self.d_goal = d_goal

        self.mean_factor = 10
        self.std_factor = 5

    def expand_goals(self, x, segment_mask):
        B, seq_len, _ = x.shape

        for b in range(B):
            goal = x[b][0]
            for l in torch.arange(seq_len)[:-1]:
                if segment_mask[b][l]:
                    goal = x[b][l + 1]
                x[b][l + 1] = goal
        return x

    def nanstd(self, o):
        return torch.sqrt(
            torch.nanmean(
                torch.pow(torch.abs(o - torch.nanmean(o)), 2))
        )

    def forward(self, x, critic_mask):
        x = self.core(x)
        # Add noise if exploration
        # Select only Segment Goals, goals between segments are discarded

        if self.exploration:
            std =self.nanstd(x)
            mean = torch.nanmean(x)
            std /= self.std_factor
            mean /= self.mean_factor
            std = std.detach()
            mean = mean.detach()
            noise = (torch.empty(self.d_goal).normal_(mean=mean, std=std).to(self.device) - (0.5 * mean)).to(
                self.device)
            x = x + noise

        x = self.expand_goals(x, critic_mask)
        return x

class WorkerCore(nn.Module):
    def __init__(self, voc_size, d_in, d_goal) -> None:
        super(WorkerCore, self).__init__()
        self.projection = nn.Linear(in_features=d_in + d_goal, out_features=voc_size)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.projection(x)
        x = self.logsoftmax(x)
        return x

class Worker(nn.Module):
    def __init__(self, voc_size, d_in, d_goal, dout_p, d_model, core=None) -> None:
        super(Worker, self).__init__()
        heads = 2
        if core is not None:
            self.core = core
        else:
            self.core = WorkerCore(voc_size, d_in, d_goal)
        self.goal_attention = MultiheadedAttention(d_goal, d_in, d_in, heads, dout_p, d_model)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, goal, mask):
        goal_completion = self.goal_attention(goal, x, x, mask)
        if not self.core.__class__ == torch.nn.Identity:
            x = torch.cat([x, goal_completion], dim=-1)
            x = self.core(x)
            #x = self.logsoftmax(x)
            return x
        return goal_completion



class BMHrlAgent(nn.Module):
    def __init__(self, cfg, train_dataset):
        super(BMHrlAgent, self).__init__()
        self.name = "bm_hrl_agent"
        self.d_video = cfg.d_vid
        self.d_audio = cfg.d_aud
        self.d_proj = cfg.rl_projection_d
        self.d_model_caps = cfg.d_model_caps
        self.d_model = cfg.d_model
        self.att_heads = cfg.rl_att_heads
        self.att_layers = cfg.rl_att_layers
        self.dout_p = cfg.dout_p
        self.d_goal = cfg.rl_goal_d
        self.voc_size = train_dataset.trg_voc_size
        self.device = get_device(cfg)

        self.critic_score_threshhold = cfg.rl_critic_score_threshhold

        self.pos_enc_A = PositionalEncoder(cfg.d_model_audio, cfg.dout_p)
        self.pos_enc_V = PositionalEncoder(cfg.d_model_video, cfg.dout_p)
        self.pos_enc_C = PositionalEncoder(cfg.d_model_caps, cfg.dout_p)

        self.critic = SegmentCritic(cfg)

        self.emb_C = VocabularyEmbedder(train_dataset.trg_voc_size, cfg.d_model_caps)
        print(self.voc_size, 'voc_size', train_dataset.train_vocab.vectors)
        self.emb_C.init_word_embeddings(train_dataset.train_vocab.vectors, cfg.unfreeze_word_emb)

        self.bm_enc = BMEncoder(d_model_M1=self.d_video, d_model_M2=self.d_audio, d_model=self.d_model,
                                d_ff_M1=cfg.rl_ff_v, d_ff_M2=cfg.rl_ff_a, dout_p=self.dout_p, H=self.att_heads,
                                N=self.att_layers)

        self.bm_worker_fus = BMFusion(
            cfg.d_model_audio, cfg.d_model_video, cfg.d_model_caps, cfg.d_model, cfg.rl_ff_c, self.dout_p,
            self.att_heads, self.att_layers
        )

        self.bm_manager_fus = BMFusion(
            cfg.d_model_audio, cfg.d_model_video, cfg.d_model_caps, cfg.d_model, cfg.rl_ff_c, self.dout_p,
            self.att_heads, self.att_layers
        )
        self.manager_core = LinearCore(cfg.d_model_caps, cfg.rl_goal_d, cfg.dout_p)

        self.manager = Manager(self.device, self.d_model_caps, self.d_goal, self.dout_p, self.manager_core)
        self.worker = Worker(voc_size=self.voc_size, d_in=self.d_model_caps, d_goal=self.d_goal, dout_p=self.dout_p,
                               d_model=self.d_model)

        self.teach_warmstart()
        self.warmstarting = True
        self.teaching_worker = True

        self.sigmoid_epoch_offset = torch.tensor(-1)

        self.worker_modules = [self.bm_enc, self.bm_worker_fus, self.worker]
        self.manager_modules = [self.bm_manager_fus, self.manager]

    def save_model(self, checkpoint_dir):
        model_file_name = checkpoint_dir + f"/{self.name}.pt"
        torch.save(self.state_dict(), model_file_name)

    def load_model(self, checkpoint_dir):
        model_file_name = checkpoint_dir + f"/{self.name}.pt"
        self.load_state_dict(torch.load(model_file_name))

    def _set_worker_grad(self, enabled):
        for name, param in self.worker.named_parameters():
            param.requires_grad = enabled
        for name, param in self.bm_worker_fus.named_parameters():
            param.requires_grad = enabled

    def _set_manager_grad(self, enabled):
        for name, param in self.manager.named_parameters():
            param.requires_grad = enabled
        for name, param in self.bm_manager_fus.named_parameters():
            param.requires_grad = enabled

    def _set_module_grads(self, modules, enable):
        for module in modules:
            for name, param in module.named_parameters():
                param.requires_grad = enable

    def teach_warmstart(self):
        self.warmstarting = True
        self._set_worker_grad(True)
        self._set_manager_grad(True)

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

        self.manager.exploration = not inference


    def warmstart(self, x, trg, mask):
        prediction = self.prediction(x, trg, mask)
        return prediction

    def prediction_audio(self, x, trg, mask):
        x_video, x_audio = x
        C = self.emb_C(trg)

        A = self.pos_enc_A(x_audio)

        segments = self.critic(C)
        segment_labels = (torch.sigmoid(segments) > self.critic_score_threshhold).squeeze().int()

        C = self.pos_enc_C(C)

    def mixed_prediction(self, x, trgs, mask, mix_factor):
        x_video, x_audio = x
        y_trg, yhat_trg = trgs

        yhat_factor = mix_factor
        y_factor = 1 - mix_factor

        C = self.emb_C(y_trg) * y_factor
        C_hat = self.emb_C(yhat_trg) * yhat_factor
        C_mixed = C + C_hat

        V = self.pos_enc_V(x_video)
        A = self.pos_enc_A(x_audio)

        return self.predict_with_features(C_mixed, V, A, mask)

    def prediction(self, x, trg, mask):
        x_video, x_audio = x
        print(trg[0], 'trg')
        C = self.emb_C(trg)
        print(C[0], 'critic')
        V = self.pos_enc_V(x_video)
        A = self.pos_enc_A(x_audio)

        return self.predict_with_features(C, V, A, mask)

    def predict_with_features(self, C, V, A, mask):
        segments = self.critic(C)
        print(segments[1])
        segments = torch.sigmoid(segments)
        print(segments[1])
        segment_labels = (segments > self.critic_score_threshhold).squeeze().int()
        print(segment_labels.shape)
        C = self.pos_enc_C(C)
        # Self Att
        Va, Av = self.bm_enc((V, A), mask)
        ##
        worker_feat = self.bm_worker_fus((C, (Av, Va)), mask)
        manager_feat = self.bm_manager_fus((C, (Av, Va)), mask)
        goals = self.manager(manager_feat, segment_labels)
        pred = self.worker(worker_feat, goals, mask["C_mask"])

        return pred, worker_feat, manager_feat, goals, segment_labels

    def inference(self, x, trg, mask):
        return self.prediction(x, trg, mask)[0]

    def forward(self, x, trg, mask, factor=1):
        prediction, worker_feat, manager_feat, goal_feat, segment_labels = self.mixed_prediction(x, trg, mask,
                                                                                                 factor) if type(
            trg) is tuple else self.prediction(x, trg, mask)

        return prediction, worker_feat, manager_feat, goal_feat, segment_labels


class UnimodalAgent(nn.Module):
    def __init__(self, cfg, train_dataset, d_m1, d_ff_m1):
        super(UnimodalAgent, self).__init__()
        self.name = "unimodal_hrl_agent"
        self.d_m1 = d_m1

        self.d_proj = cfg.rl_projection_d
        self.d_model_caps = cfg.d_model_caps
        self.d_model = cfg.d_model
        self.att_heads = cfg.rl_att_heads
        self.att_layers = cfg.rl_att_layers
        self.dout_p = cfg.dout_p
        self.d_goal = cfg.rl_goal_d
        self.voc_size = train_dataset.trg_voc_size
        self.device = get_device(cfg)

        self.critic_score_threshhold = cfg.rl_critic_score_threshhold

        #
        self.pos_enc = PositionalEncoder(self.d_m1, cfg.dout_p)

        self.pos_enc_C = PositionalEncoder(cfg.d_model_caps, cfg.dout_p)

        self.critic = SegmentCritic(cfg)

        self.emb_C = VocabularyEmbedder(train_dataset.trg_voc_size, cfg.d_model_caps)
        self.emb_C.init_word_embeddings(train_dataset.train_vocab.vectors, cfg.unfreeze_word_emb)

        self.uni_enc = UnimodalEncoder(d_model_M1=self.d_m1, d_model=self.d_model, d_ff_M1=d_ff_m1, dout_p=self.dout_p,
                                       H=self.att_heads, N=self.att_layers)

        self.uni_worker_fus = UnimodalFusion(
            self.d_m1, cfg.d_model_caps, cfg.d_model, d_ff_m1, self.dout_p,
            self.att_heads, self.att_layers
        )

        self.uni_manager_fus = UnimodalFusion(
            self.d_m1, cfg.d_model_caps, cfg.d_model, d_ff_m1, self.dout_p,
            self.att_heads, self.att_layers
        )

        self.manager = Manager(self.device, self.d_model_caps, self.d_goal, self.dout_p)
        self.worker = Worker(voc_size=self.voc_size, d_in=self.d_model_caps, d_goal=self.d_goal, dout_p=self.dout_p,
                               d_model=self.d_model)

        self.teach_warmstart()
        self.warmstarting = True
        self.teaching_worker = True

        self.worker_modules = [self.uni_enc, self.uni_worker_fus, self.worker]
        self.manager_modules = [self.uni_manager_fus, self.manager]

    def save_model(self, checkpoint_dir):
        model_file_name = checkpoint_dir + f"/{self.name}.pt"
        torch.save(self.state_dict(), model_file_name)

    def load_model(self, checkpoint_dir):
        model_file_name = checkpoint_dir + f"/{self.name}.pt"
        self.load_state_dict(torch.load(model_file_name))

    def _set_worker_grad(self, enabled):
        for name, param in self.worker.named_parameters():
            param.requires_grad = enabled
        for name, param in self.uni_worker_fus.named_parameters():
            param.requires_grad = enabled

    def _set_manager_grad(self, enabled):
        for name, param in self.manager.named_parameters():
            param.requires_grad = enabled
        for name, param in self.uni_manager_fus.named_parameters():
            param.requires_grad = enabled

    def _set_module_grads(self, modules, enable):
        for module in modules:
            for name, param in module.named_parameters():
                param.requires_grad = enable

    def teach_warmstart(self):
        self.warmstarting = True
        self._set_worker_grad(True)
        self._set_manager_grad(True)

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
        else:
            self.manager.exploration = True

    def warmstart(self, x, trg, mask):
        prediction = self.prediction(x, trg, mask)
        return prediction

    def prediction(self, x, trg, mask):
        m1_mask, c_mask = mask
        C = self.emb_C(trg)

        m1 = self.pos_enc(x)

        segments = self.critic(C)
        segment_labels = (torch.sigmoid(segments) > self.critic_score_threshhold).squeeze().int()

        C = self.pos_enc_C(C)

        # Self Att
        m1 = self.uni_enc(m1, m1_mask)
        ##

        worker_feat = self.uni_worker_fus((C, m1), mask)
        manager_feat = self.uni_manager_fus((C, m1), mask)

        goals = self.manager(manager_feat, segment_labels)
        pred = self.worker(worker_feat, goals, c_mask)

        return pred, worker_feat, manager_feat, goals, segment_labels

    def inference(self, x, trg, mask):
        return self.prediction(x, trg, mask)[0]

    def forward(self, x, trg, mask):
        prediction, worker_feat, manager_feat, goal_feat, segment_labels = self.prediction(x, trg, mask)

        return prediction, worker_feat, manager_feat, goal_feat, segment_labels


class AudioAgent(UnimodalAgent):
    def __init__(self, cfg, train_dataset) -> None:
        super(AudioAgent, self).__init__(cfg, train_dataset, cfg.d_aud, cfg.rl_ff_a)


class VideoAgent(UnimodalAgent):
    def __init__(self, cfg, train_dataset) -> None:
        super(VideoAgent, self).__init__(cfg, train_dataset, cfg.d_vid, cfg.rl_ff_v)
