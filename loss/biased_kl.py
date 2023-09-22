import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


def test_print(msg):
    print(msg, file=sys.stderr)


class BiasedKL(nn.Module):
    def __init__(self, label_smoothing, pad_idx):
        super(BiasedKL, self).__init__()

        self.pad_idx = pad_idx
        self.ls = label_smoothing
        self.trg_factor = 1 - self.ls
        self.kl = nn.KLDivLoss(reduction="none")
        self.eps = 1e-5

    def forward(self, pred, trg, biased_trg, biased_offset):
        B, S, V = pred.shape
        trg_ampl = self.trg_factor * (1 - biased_offset).contiguous().view(-1)
        normed_offset = biased_offset * self.trg_factor
        biased_dist = torch.zeros_like(pred)
        biased_dist = torch.scatter(biased_dist, 2, biased_trg, normed_offset)

        # (B, S, V) -> (B * S, V); (B, S) -> (B * S)
        prediction = pred.contiguous().view(-1, V)
        target = trg.contiguous().view(-1)

        test_print(f'{target.unsqueeze(-1).shape}, {trg_ampl.unsqueeze(-1).shape}')

        # prior (uniform)
        dist = self.ls * torch.ones_like(prediction) / (V - 2)
        # add smoothed ground-truth to prior (args: dim, index, src (value))
        rep_trg = torch.transpose(target.long().repeat(biased_trg.shape[-1]).reshape(-1, B * S), 0, 1)
        re_trg_amp = torch.transpose(trg_ampl.reshape(-1, B * S), 0, 1)
        dist.scatter_(1, rep_trg,
                      re_trg_amp)  # Essentially "One Hot" encode target with .3 (rest is 1/vocsize-1 * .7)
        # make the padding token to have zero probability
        dist[:, self.pad_idx] = 0
        dist = dist + biased_dist.contiguous().view(-1, V)
        # ?? mask: 1 if target == pad_idx; 0 otherwise
        mask = torch.nonzero(target == self.pad_idx)

        if mask.sum() > 0 and len(mask) > 0:  # (padded sentences are present)
            # dim, index, val
            dist.index_fill_(0, mask.squeeze(), 0)  # set distance 0 where there are padding tokens
        divergence = self.kl(prediction, dist + self.eps)
        return divergence


class Reinforce(nn.Module):
    def __init__(self):
        super(Reinforce, self).__init__()
        self.pad_idx = 0
        self.value_const = 0.5
        self.entropie_const = 0.005
        self.eps = 1e-5

    def forward(self, pred, action_probs, value, critic_value, segments=None, manager_policy = None):
        B, L, V = pred.shape
        #policy_action = torch.sum(action_probs, - 1)

        advantage = value - critic_value
        #print(policy_action.shape, advantage.shape)
        assert (segments is None and manager_policy is None) or (not segments is None and not manager_policy is None)
        if segments is None:
            policy_loss = -torch.mean(advantage.clone().detach().squeeze() * (torch.log(action_probs)))
            value_loss = torch.mean(torch.pow(advantage, 2))
            entropy = -1.0 * torch.sum(pred * torch.log(pred), -1)
            entropy_loss = -1.0 * torch.mean(entropy)
            loss = policy_loss + self.value_const * value_loss + entropy_loss * self.entropie_const
            return loss
        else:
            segment_prob = torch.zeros(B, L, dtype=torch.float32)
            segment_count = torch.zeros(B, dtype=torch.int)
            segment_indices = torch.nonzero(segments)
            old_l = old_b = 0
            for segment_idx in segment_indices:
                b, l = segment_idx
                if b != old_b:
                    segment_prob[old_b, old_l:] = torch.sum(torch.log(action_probs[old_b, old_l:]))
                    old_b = b
                    old_l = 0
                segment_prob[b, old_l:l + 1] = torch.sum(torch.log(action_probs[b, old_l:l + 1]))
                old_l = l + 1
                segment_count[b] += 1
            log_sum_action_probs = segment_prob.to(pred.get_device())
            policy_loss = -torch.mean(advantage.clone().detach().squeeze() * log_sum_action_probs)
            value_loss = torch.mean(torch.pow(advantage, 2))
            entropy = -1.0 * torch.sum(pred * torch.log(pred), -1)
            entropy_loss = -1.0 * torch.mean(entropy)
            loss = policy_loss + self.value_const * value_loss + entropy_loss * self.entropie_const
            return loss



