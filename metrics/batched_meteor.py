from nltk.translate.meteor_score import meteor_score
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.translate import meteor
from nltk.translate.meteor_score import single_meteor_score
import nltk
import sys
from .util import cook_test, cook_refs, precook, discontinue_reward
import torch
import numpy as np



def get_gamma_matrix(gamma, B, L):
    # start at 1
    gamma_exp = torch.arange(L, dtype=torch.float32).repeat((B, 1))
    gamma_mat = gamma ** gamma_exp
    return gamma_mat

def segment_reward(reward, sections):
    B, L = reward.shape
    segment_reward = torch.zeros(B, L, dtype=torch.float32)
    segment_count = torch.zeros(B, dtype=torch.int)

    segment_indices = torch.nonzero(sections)

    for segment_idx in segment_indices:
        b, l = segment_idx
        reward_idx = segment_count[b]
        segment_reward[b, reward_idx] = reward[b, l]
        segment_count[b] += 1
    return segment_reward, segment_indices

def expand_gamma(gamma):
    B, L = gamma.shape
    gammas = torch.zeros(B, L, L)
    for j in (np.arange(L) + 1)[::-1]:
        index = L - j
        prefix = torch.zeros(L - j, dtype=torch.float32).repeat((B, 1))
        res = torch.cat((prefix, gamma[:, :j]), 1)

        gammas[:, index, :] = res
    return gammas


def test_print(msg):
    print(msg, file=sys.stderr)


def word_from_vector(vocab, indices):
    return np.array([vocab.itos[i] for i in indices])


class MeteorScorer():
    type = "METEOR"
    #print('downloaded wordnet')
    #nltk.download('wordnet')

    def _meteor_diff(self, pred, trg, mask):
        last_token = (torch.sum(mask, dim=1))
        B, L = pred.shape
        rewards = torch.zeros(B, L, dtype=torch.float32)

        for b in torch.arange(B):
            seq_len = last_token[b] - 1  # TODO remove
            hypo = word_from_vector(self.vocab, pred[b])

            for l in torch.arange(seq_len):
                partial_hypo = " ".join(hypo[:l + 1])
                # reward = meteor(trg[b], partial_hypo
        while True:
            try:
                reward = single_meteor_score(trg[b], partial_hypo)
                break
            except LookupError:
                nltk.download('wordnet')
        rewards[b, l] = reward  # meteor_score([trg[b]], self.detokenizer.detokenize(partial_hypo))
                # TODO try also cutting ref to match hypo? but could overfit

        delta_meteor = rewards[:, 1:] - rewards[:, :-1]
        # reward not seen by diff at pos 0
        delta_meteor = torch.cat((rewards[:, 0].unsqueeze(-1), delta_meteor), dim=1).to(self.device)
        # Account for the shifted value thats outside the mask now
        # delta_meteor *= mask.float()
        return delta_meteor, rewards

    def delta_meteor_segment(self, delta_meteor_step_reward, sections, gamma):
        segment_meteor_dif, segment_reward_index = segment_reward(delta_meteor_step_reward, sections)
        discounted_segment_reward = discontinue_reward(segment_meteor_dif, gamma)
        return discounted_segment_reward, segment_reward_index

    def delta_meteor_step(self, pred, trg, mask, gamma):
        meteor_diff, rewards = self._meteor_diff(pred, trg, mask)
        meteor_diff = meteor_diff.to(self.device)
        result = discontinue_reward(meteor_diff.to(self.device), gamma)
        # print(result)
        # discounted_cider= torch.einsum("bl,bsl->bs", cider_diff, gamma_matrix)
        return result, rewards

    def delta_meteor_worker(self, pred, trg, mask):
        delta_meteor_step_reward, rewards = self.delta_meteor_step(pred, trg, mask, self.gamma)
        return torch.tensor(delta_meteor_step_reward).float(), rewards

    def delta_meteor_manager(self, pred, trg, mask, sections):
        manager_segment_score, rewards = self.delta_meteor(pred, trg, mask, sections)
        return torch.tensor(manager_segment_score).float(), None

    def delta_meteor(self, pred, trg, mask, sections):
        delta_meteor_step_reward, rewards = self.delta_meteor_step(pred, trg, mask, self.gamma)
        sections[:, 0] = 1  # Set section delimiter so first sections doesnt disappear
        #Above is in memory manipulation, the original tensor will havge its entries modified, shouldnt matter but could in further computations
        delta_meteor_section_reward, segment_idx = self.delta_meteor_segment(torch.tensor(delta_meteor_step_reward),
                                                                           sections, self.gamma)
        bool_mask = sections.bool()
        segment_n_per_sentence = torch.sum(sections, dim=1)  # Set section delimiter so first sections doesnt disappear
        values_flat = None
        for row, n in enumerate(segment_n_per_sentence):
            value = delta_meteor_section_reward[row, :n].to(self.device)
            if values_flat is None:
                values_flat = value
            else:
                values_flat = torch.cat((values_flat, value), dim=0).to(self.device)
        B, L = delta_meteor_section_reward.shape
        final_reward = torch.zeros(B, L, dtype=torch.float32).to(self.device)
        final_reward[bool_mask] = values_flat.float()
        return final_reward, rewards

    def get_meteor_gamma(self, gamma, B, L):
        gamma_mat = get_gamma_matrix(gamma, B, L)
        return expand_gamma(gamma_mat).to(self.device)

    def __init__(self, vocab, device, gamma_step, gamma_section) -> None:
        self.vocab = vocab
        self.device = device
        self.gamma = gamma_step
        self.gamma_manager= gamma_section
        self.detokenizer = TreebankWordDetokenizer()
        # TODO use different gamm for worker/manager
