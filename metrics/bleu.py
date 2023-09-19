import copy

import torch.nn.functional as F
import numpy as np
import torch
import math
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from .batched_meteor import word_from_vector, expand_gamma, get_gamma_matrix, segment_reward
from .util import cook_test_bleu, cook_refs_bleu, precook, discontinue_reward


class BleuScorer():
    def __init__(self, vocab, device, gamma, gamma_manager, n=4, sigma=6.0,):
        # set blue to sum over 1 to 4-grams
        assert(n <= 4 and n > 0)
        self.counter = 0
        self.vocab = vocab
        self.device = device
        self._n = n
        # set the standard deviation parameter for gaussian penalty
        self.type = "BLEU"
        self.gamma = gamma
        self.gamma_m = gamma_manager
        self.bleu_scorer = BleuScorerObj(n=self._n)
    def _bleu_diff(self, pred, target):
        B, L = pred.shape

        rewards = None
        for b in torch.arange(B):
            hypo = list(word_from_vector(self.vocab, pred[b]))
            #hypo = target[b].split() #TODO remove this when no longer in development
            # hypo.append('bug')
            res = [target[b].lower()]
            #print(res, hypo)
            scores = []
            for l, _ in enumerate(hypo):
                partial_hypo = " ".join(hypo[:l + 1]).lower()

                self.bleu_scorer += (partial_hypo, res)
                (score, test) = self.bleu_scorer.compute_score()
                scores.append(score)
                self.bleu_scorer.reset_bleu_scorer()
            #print(scores)
            scores = torch.tensor(scores).to(self.device)
            pad_dim = L - scores.shape[0]
            hypo_len = len(hypo)
            scores = F.pad(scores, [0, pad_dim], "constant", scores[hypo_len - 1]).to(self.device)
            scores = torch.reshape(scores, (1, -1)).to(self.device)
            if rewards is None:
                rewards = scores
            else:
                rewards = torch.cat((rewards, scores), dim=0).to(self.device)
        rewards = rewards
        delta_bleu = rewards[:, 1:] - rewards[:, :-1]
        delta_bleu = torch.cat((rewards[:, 0].unsqueeze(-1), delta_bleu), dim=1).to(self.device)
        self.counter += 1

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

        delta_bleu_section_reward, segment_idx = self.delta_bleu_segment(torch.tensor(delta_bleu_step_reward), sections, self.gamma)


        return delta_bleu_section_reward, rewards

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

    __slots__ = "n", "crefs", "ctest", "_score", "_ratio", "_testlen", "_reflen", "special_reflen"

    # special_reflen is used in oracle (proportional effective ref len for a node).

    def copy(self):
        ''' copy the refs.'''
        new = BleuScorer(n=self.n)
        new.ctest = copy.copy(self.ctest)
        new.crefs = copy.copy(self.crefs)
        new._score = None
        return new

    def __init__(self, test=None, refs=None, n=4, special_reflen=None):
        ''' singular instance '''

        self.n = n
        self.crefs = []
        self.ctest = []
        self.cook_append(test, refs)
        self.special_reflen = special_reflen

    def cook_append(self, test, refs):
        '''called by constructor and __iadd__ to avoid creating new instances.'''

        if refs is not None:
            self.crefs.append(cook_refs_bleu(refs))
            if test is not None:
                cooked_test = cook_test_bleu(test, self.crefs[-1])
                self.ctest.append(cooked_test)  ## N.B.: -1
            else:
                self.ctest.append(None)  # lens of crefs and ctest have to match

        self._score = None  ## need to recompute

    def ratio(self, option=None):
        self.compute_score(option=option)
        return self._ratio

    def score_ratio(self, option=None):
        '''return (bleu, len_ratio) pair'''
        return (self.fscore(option=option), self.ratio(option=option))

    def score_ratio_str(self, option=None):
        return "%.4f (%.2f)" % self.score_ratio(option)

    def reflen(self, option=None):
        self.compute_score(option=option)
        return self._reflen

    def testlen(self, option=None):
        self.compute_score(option=option)
        return self._testlen

    def retest(self, new_test):
        if type(new_test) is str:
            new_test = [new_test]
        assert len(new_test) == len(self.crefs), new_test
        self.ctest = []
        for t, rs in zip(new_test, self.crefs):
            self.ctest.append(cook_test_bleu(t, rs))
        self._score = None

        return self

    def rescore(self, new_test):
        ''' replace test(s) with new test(s), and returns the new score.'''

        return self.retest(new_test).compute_score()

    def size(self):
        assert len(self.crefs) == len(self.ctest), "refs/test mismatch! %d<>%d" % (len(self.crefs), len(self.ctest))
        return len(self.crefs)

    def __iadd__(self, other):
        '''add an instance (e.g., from another sentence).'''

        if type(other) is tuple:
            ## avoid creating new BleuScorer instances
            self.cook_append(other[0], other[1])
        else:
            assert self.compatible(other), "incompatible BLEUs."
            self.ctest.extend(other.ctest)
            self.crefs.extend(other.crefs)
            self._score = None  ## need to recompute

        return self

    def compatible(self, other):
        return isinstance(other, BleuScorer) and self.n == other.n

    def reset_bleu_scorer(self):
        self.crefs = []
        self.ctest = []
        self._reflen = None
        self._score = None

    def single_reflen(self, option="average"):
        return self._single_reflen(self.crefs[0][0], option)

    def _single_reflen(self, reflens, option=None, testlen=None):

        if option == "shortest":
            reflen = min(reflens)
        elif option == "average":
            reflen = float(sum(reflens)) / len(reflens)
        elif option == "closest":
            reflen = min((abs(l - testlen), l) for l in reflens)[1]
        else:
            assert False, "unsupported reflen option %s" % option

        return reflen

    def recompute_score(self, option=None, verbose=0):
        self._score = None
        return self.compute_score(option, verbose)

    def compute_score(self, option=None, verbose=0):
        n = self.n
        small = 1e-9
        tiny = 1e-15  ## so that if guess is 0 still return 0
        bleu_list = [[] for _ in range(n)]

        if self._score is not None:
            return self._score

        if option is None:
            option = "average" if len(self.crefs) == 1 else "closest"

        self._testlen = 0
        self._reflen = 0
        totalcomps = {'testlen': 0, 'reflen': 0, 'guess': [0] * n, 'correct': [0] * n}

        # for each sentence
        for comps in self.ctest:
            testlen = comps['testlen']
            self._testlen += testlen

            if self.special_reflen is None:  ## need computation
                reflen = self._single_reflen(comps['reflen'], option, testlen)
            else:
                reflen = self.special_reflen

            self._reflen += reflen

            for key in ['guess', 'correct']:
                for k in range(n):
                    totalcomps[key][k] += comps[key][k]

            # append per image bleu score
            bleu = 1.
            for k in range(n):
                bleu *= (float(comps['correct'][k]) + tiny) \
                        / (float(comps['guess'][k]) + small)
                bleu_list[k].append(bleu ** (1. / (k + 1)))
            ratio = (testlen + tiny) / (reflen + small)  ## N.B.: avoid zero division
            if ratio < 1:
                for k in range(n):
                    bleu_list[k][-1] *= math.exp(1 - 1 / ratio)
            if verbose > 1:
                print(comps, reflen)

        totalcomps['reflen'] = self._reflen
        totalcomps['testlen'] = self._testlen
        bleus = []
        bleu = 1.
        for k in range(n):
            bleu *= float(totalcomps['correct'][k] + tiny) \
                    / (totalcomps['guess'][k] + small)
            bleus.append(bleu ** (1. / (k + 1)))
        ratio = (self._testlen + tiny) / (self._reflen + small)  ## N.B.: avoid zero division
        if ratio < 1:
            for k in range(n):
                bleus[k] *= math.exp(1 - 1 / ratio)

        if verbose > 0:
            print(totalcomps)
            print("ratio:", ratio)

        self._score = bleus
        bleu_weights = 1/n
        avg_score = torch.sum(torch.tensor(self._score) * torch.tensor(bleu_weights))
        return avg_score, self._score