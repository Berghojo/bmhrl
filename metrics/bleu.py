import copy

import torch.nn.functional as F
import numpy as np
import torch
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from .batched_meteor import word_from_vector, expand_gamma, get_gamma_matrix, segment_reward
from .util import cook_test, cook_refs, precook, discontinue_reward


class BleuScorer():
    def __init__(self, vocab, dictionary, device, gamma, gamma_manager, n=4, sigma=6.0,):
        # set blue to sum over 1 to 4-grams
        assert(n <= 4 and n > 0)
        self.counter = 0
        self.vocab = vocab
        self.device = device
        self._n = n
        # set the standard deviation parameter for gaussian penalty
        self._sigma = sigma
        self.type = "BLEU"
        self.gamma = gamma
        self.gamma_m = gamma_manager
        self.dictionary = dictionary

    def _bleu_diff(self, pred, target):
        B, L = pred.shape
        Bleu = BleuScorerObj(self.dictionary, n=self._n, sigma=self._sigma)
        rewards = None
        for b in torch.arange(B):
            hypo = list(word_from_vector(self.vocab, pred[b]))
            #hypo = target[b].split() #TODO remove this when no longer in development

            for l, _ in enumerate(hypo):
                partial_hypo = " ".join(hypo[:l + 1])
                res = target[b].split()
                bleu_scorer += (partial_hypo, res)

            (_, scores) = bleu_scorer.compute_score()
            scores = torch.tensor(scores).to(self.device)
            pad_dim = L - scores.shape[0]
            hypo_len = len(hypo)
            scores = F.pad(scores, [0, pad_dim], "constant", scores[hypo_len - 1]).to(self.device)
            #TODO Check if padding needs to set to 0 or the last value of the hyphon
            bleu_scorer.reset_bleu_scorer()
            scores = torch.reshape(scores, (1, -1)).to(self.device)
            if rewards is None:
                rewards = scores
            else:
                rewards = torch.cat((rewards, scores), dim=0).to(self.device)
        delta_bleu = rewards[:, 1:] - rewards[:, :-1]
        delta_bleu = torch.cat((rewards[:, 0].unsqueeze(-1), delta_bleu), dim=1).to(self.device)
        self.counter += 1
        rewards = None
        return delta_bleu.float(), rewards

    def delta_bleu_manager(self, pred, trg, mask, sections):
        manager_segment_score, rewards = self.delta_bleu(pred, trg, mask, sections)
        return torch.tensor(manager_segment_score).float(), None

    def delta_bleu_worker(self, pred, trg):
        #gamma_matrix = get_gamma_matrix(self.gamma, B, L)

        delta_bleu_step_reward, rewards = self.delta_bleu_step(pred, trg, self.gamma)
        return torch.tensor(delta_bleu_step_reward).float(), rewards


    def delta_bleu(self, pred, trg, mask, sections):
        delta_bleu_step_reward, rewards = self.delta_bleu_step(pred, trg, self.gamma)
        sections[:, 0] = 1  # Set section delimiter so first sections doesnt disappear
        # TODO Above is in memory manipulation, the original tensor will havge its entries modified, shouldnt matter but could in further computations
        delta_bleu_section_reward, segment_idx = self.delta_bleu_segment(torch.tensor(delta_bleu_step_reward), sections, self.gamma)
        bool_mask = sections.bool()
        segment_n_per_sentence = torch.sum(sections, dim=1) # Set section delimiter so first sections doesnt disappear
        values_flat = None
        for row, n in enumerate(segment_n_per_sentence):
            value = delta_bleu_section_reward[row, :n].to(self.device)
            if values_flat is None:
                values_flat = value
            else:
                values_flat = torch.cat((values_flat, value), dim=0).to(self.device)
        B, L = delta_bleu_section_reward.shape
        final_reward = torch.zeros(B, L, dtype=torch.float32).to(self.device)
        final_reward[bool_mask] = values_flat.float()
        return final_reward, rewards

    def delta_bleu_segment(self, delta_bleu_step_reward, sections, gamma):
        segment_bleu_dif, segment_reward_index = segment_reward(delta_bleu_step_reward, sections)
        discounted_segment_reward = discontinue_reward(segment_bleu_dif, gamma)
        return discounted_segment_reward, segment_reward_index

    def delta_bleu_step(self, pred, tar, gamma):
        bleu_diff, rewards = self._bleu_diff(pred, tar)
        bleu_diff = bleu_diff.to(self.device)
        result = discontinue_reward(bleu_diff, gamma)
        #discounted_cider= torch.einsum("bl,bsl->bs", cider_diff, gamma_matrix)
        return result, rewards



class BleuScorerObj(object):
    """Bleu scorer.
    """

    def copy(self):
        ''' copy the refs.'''
        new = BleuScorer(n=self.n)
        new.ctest = copy.copy(self.ctest)
        new.crefs = copy.copy(self.crefs)
        return new

    def __init__(self, doc_frequency, test=None, refs=None, n=4, sigma=6.0):
        ''' singular instance '''
        self.n = n
        self.sigma = sigma
        self.crefs = []
        self.ctest = []
        self.document_frequency = doc_frequency
        self.cook_append(test, refs)
        self.ref_len = None

    def compute_average_bleu_over_dataset(model, dataloader, target_sos, target_eos, device):
        '''Determine the average BLEU score across sequences
        '''
        with torch.no_grad():
            total_score = [0, 0, 0, 0]
            total_num = 0
            for data in tqdm(dataloader):
                torch.cuda.empty_cache()
                images, captions_ref, cap_lens = data
                captions_ref = pad_sequence(captions_ref, padding_value=target_eos)
                images = images.to(device)
                total_num += len(cap_lens)
                b_1 = model(images, on_max='halt')
                captions_cand = b_1[..., 0]
                batch_score = compute_batch_total_bleu(captions_ref, captions_cand, target_sos, target_eos)
                total_score = [total_score[i] + batch_score[i] for i in range(len(total_score))]

            total_score = [total_score[i] / total_num for i in range(len(total_score))]
            return total_score

    def compute_batch_total_bleu(captions_ref, captions_cand, target_sos, target_eos):
        '''Compute the total BLEU score over elements in a batch
        '''
        with torch.no_grad():
            refs = captions_ref.T
            cands = captions_cand.T
            refs_list = refs.tolist()
            cands_list = cands.tolist()
            for i in range(len(refs_list)):  # Removes sos tags
                refs_list[i] = list(filter((target_sos).__ne__, refs_list[i]))
                cands_list[i] = list(filter((target_sos).__ne__, cands_list[i]))

            for i in range(len(refs_list)):  # Removes eos tags
                refs_list[i] = list(filter((target_eos).__ne__, refs_list[i]))
                cands_list[i] = list(filter((target_eos).__ne__, cands_list[i]))

            total_bleu_scores = [0, 0, 0, 0]
            for i in range(refs.shape[0]):
                ref = refs_list[i]
                cand = cands_list[i]
                for n in range(len(total_bleu_scores)):
                    score = BLEU_score(ref, cand, n + 1)
                    total_bleu_scores[n] += score
            return total_bleu_scores

    def grouper(seq, n):
        '''Extract all n-grams from a sequence
        '''
        ngrams = []
        for i in range(len(seq) - n + 1):
            ngrams.append(seq[i:i + n])

        return ngrams

    def n_gram_precision(reference, candidate, n):
        '''Calculate the precision for a given order of n-gram
        '''
        total_matches = 0
        ngrams_r = grouper(reference, n)
        ngrams_c = grouper(candidate, n)
        total_num = len(ngrams_c)
        assert total_num > 0
        for ngram_c in ngrams_c:
            if ngram_c in ngrams_r:
                total_matches += 1
        return total_matches / total_num

    def brevity_penalty(reference, candidate):
        '''Calculate the brevity penalty between a reference and candidate
        '''
        if len(candidate) == 0:
            return 0
        if len(reference) <= len(candidate):
            return 1
        return np.exp(1 - (len(reference) / len(candidate)))

    def BLEU_score(reference, hypothesis, n):
        '''Calculate the BLEU score
        '''
        bp = brevity_penalty(reference, hypothesis)
        prec = 1
        cand_len = min(n, len(hypothesis))
        if (cand_len == 0):
            return 0
        for i in range(1, cand_len + 1):
            prec = prec * n_gram_precision(reference, hypothesis, i)
        prec = prec ** (1 / n)
        return bp * prec