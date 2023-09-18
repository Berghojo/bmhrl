import copy
from collections import defaultdict
import numpy as np
import math
import torch.nn.functional as F
import torch
from .batched_meteor import word_from_vector, expand_gamma, get_gamma_matrix, segment_reward
from .util import cook_test, cook_refs, precook, discontinue_reward


class CiderScorer():

    def __init__(self, vocab, dictionary, device, gamma, gamma_manager, n=4, sigma=6.0,):
        # set cider to sum over 1 to 4-grams
        assert(n <= 4 and n > 0)
        self.counter = 0
        self.vocab = vocab
        self.device = device
        self._n = n
        # set the standard deviation parameter for gaussian penalty
        self._sigma = sigma
        self.type = "CIDER"
        self.gamma = gamma
        self.gamma_m = gamma_manager
        self.dictionary = precook_corpus(dictionary)

    def _cider_diff(self, pred, target):
        B, L = pred.shape
        cider_scorer = CiderScorerObj(self.dictionary, n=self._n, sigma=self._sigma)
        rewards = None
        for b in torch.arange(B):
            hypo = list(word_from_vector(self.vocab, pred[b]))
            #hypo = target[b].lower().split() #TODO remove this when no longer in development
            scores = []
            res = target[b].lower()
            for l, _ in enumerate(hypo):
                partial_hypo = " ".join(hypo[:l + 1])

                cider_scorer += (partial_hypo, res)
                (_, score) = cider_scorer.compute_score()

                scores.append(score[0])
                cider_scorer.reset_cider_scorer()
            scores = torch.tensor(scores).to(self.device)

            pad_dim = L - scores.shape[0]
            hypo_len = len(hypo)
            scores = F.pad(scores, [0, pad_dim], "constant", scores[hypo_len - 1]).to(self.device)

            scores = torch.reshape(scores, (1, -1)).to(self.device)
            if rewards is None:
                rewards = scores
            else:
                rewards = torch.cat((rewards, scores), dim=0).to(self.device)
        delta_cider = rewards[:, 1:] - rewards[:, :-1]
        delta_cider = torch.cat((rewards[:, 0].unsqueeze(-1), delta_cider), dim=1).to(self.device)
        self.counter += 1

        return delta_cider.float(), rewards

    def delta_cider_manager(self, pred, trg, mask, sections):
        manager_segment_score, rewards = self.delta_cider(pred, trg, mask, sections)

        return torch.tensor(manager_segment_score).float(), None

    def delta_cider_worker(self, pred, trg):
        #gamma_matrix = get_gamma_matrix(self.gamma, B, L)

        delta_cider_step_reward, rewards = self.delta_cider_step(pred, trg, self.gamma)

        return delta_cider_step_reward.clone().detach().float(), rewards


    def delta_cider(self, pred, trg, mask, sections):
        words_per_sent = mask.sum(dim=-1) - 1
        for n, i in enumerate(list(words_per_sent)):
            sections[n, i] = 1
        sections[~mask] = 0
        segm_per_row = sections.sum(dim=-1)
        delta_cider_step_reward, rewards = self.delta_cider_step(pred, trg, self.gamma)
        delta_cider_section_reward, segment_idx = self.delta_cider_segment(torch.tensor(delta_cider_step_reward), sections, self.gamma)
        bool_mask = sections.bool()
        values_flat = torch.cat([delta_cider_section_reward[row, :elements]
                                 for row, elements in enumerate(list(segm_per_row))], dim=0)

        values_flat = values_flat.to(self.device)
        B, L = delta_cider_section_reward.shape
        final_reward = torch.zeros(B, L, dtype=torch.float32).to(self.device)
        final_reward[bool_mask] = values_flat.float()
        return final_reward, rewards

    def delta_cider_segment(self, delta_cider_step_reward, sections, gamma):
        segment_cider_dif, segment_reward_index = segment_reward(delta_cider_step_reward, sections)
        discounted_segment_reward = discontinue_reward(segment_cider_dif, gamma)
        return discounted_segment_reward, segment_reward_index

    def delta_cider_step(self, pred, tar, gamma):
        cider_diff, rewards = self._cider_diff(pred, tar)
        cider_diff = cider_diff.to(self.device)
        result = discontinue_reward(cider_diff, gamma)

        return result, rewards


def precook_corpus(caps, n=4, out=False):
    counts = defaultdict(int)
    for cap in caps:
        for k in range(1, n+1):
            for i in range(len(cap)-k+1):
                ngram = tuple(cap[i:i+k])
                counts[ngram] += 1
    return defaultdict(int, {key:val for key, val in counts.items() if val > 1})
class CiderScorerObj(object):
    """CIDEr scorer.
    """

    def copy(self):
        ''' copy the refs.'''
        new = CiderScorer(n=self.n)
        new.ctest = copy.copy(self.ctest)
        new.crefs = copy.copy(self.crefs)
        return new

    def __init__(self, doc_frequency, test=None, refs=None, n=4, sigma=6.0):
        ''' singular instance '''
        self.n = n
        self.sigma = sigma
        self.crefs = []
        self.ctest = []
        #self.document_frequency = doc_frequency

        self.document_frequency = doc_frequency
        self.ref_len = None

    def cook_append(self, test, refs):
        '''called by constructor and __iadd__ to avoid creating new instances.'''

        if refs is not None:
            self.crefs.append(cook_refs([refs], n=self.n))
            if test is not None:
                self.ctest.append(cook_test(test, n=self.n)) ## N.B.: -1
            else:
                self.ctest.append(None) # lens of crefs and ctest have to match#

    def size(self):
        assert len(self.crefs) == len(self.ctest), "refs/test mismatch! %d<>%d" % (len(self.crefs), len(self.ctest))
        return len(self.crefs)

    def __iadd__(self, other):
        '''add an instance (e.g., from another sentence).'''

        if type(other) is tuple:
            ## avoid creating new CiderScorer instances
            self.cook_append(other[0], other[1])
        else:
            self.ctest.extend(other.ctest)
            self.crefs.extend(other.crefs)

        return self
    def reset_cider_scorer(self):
        self.crefs = []
        self.ctest = []
        self.ref_len = None

    def compute_cider(self):
        def counts2vec(cnts, is_ref=False):
            """
            Function maps counts of ngram to vector of tfidf weights.
            The function returns vec, an array of dictionary that store mapping of n-gram and tf-idf weights.
            The n-th entry of array denotes length of n-grams.
            :param cnts:
            :return: vec (array of dict), norm (array of float), length (int)
            """
            vec = [defaultdict(float) for _ in range(self.n)]
            length = 0
            norm = [0.0 for _ in range(self.n)]
            for (ngram, term_freq) in cnts.items():

                # give word count 1 if it doesn't appear in reference corpus
                df = np.log(max(1.0, self.document_frequency[ngram]))
                # ngram index
                n = len(ngram)-1
                # tf (term_freq) * idf (precomputed idf) for n-grams
                vec[n][ngram] = float(term_freq)*(self.ref_len - df)
                # compute norm for the vector.  the norm will be used for computing similarity
                norm[n] += pow(vec[n][ngram], 2)

                if n == 1:
                    length += term_freq
            norm = [np.sqrt(n) for n in norm]
            return vec, norm, length

        def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
            '''
            Compute the cosine similarity of two vectors.
            :param vec_hyp: array of dictionary for vector corresponding to hypothesis
            :param vec_ref: array of dictionary for vector corresponding to reference
            :param norm_hyp: array of float for vector corresponding to hypothesis
            :param norm_ref: array of float for vector corresponding to reference
            :param length_hyp: int containing length of hypothesis
            :param length_ref: int containing length of reference
            :return: array of score for each n-grams cosine similarity
            '''
            delta = float(length_hyp - length_ref)
            # measure consine similarity
            val = np.array([0.0 for _ in range(self.n)])
            for n in range(self.n):
                for (ngram, count) in vec_hyp[n].items():
                    val[n] += min(vec_hyp[n][ngram], vec_ref[n][ngram]) * vec_ref[n][ngram]
                if (norm_hyp[n] != 0) and (norm_ref[n] != 0):
                    val[n] /= (norm_hyp[n]*norm_ref[n])

                assert(not math.isnan(val[n]))
                # vrama91: added a length based gaussian penalty
                val[n] *= np.e**(-(delta**2)/(2*self.sigma**2))

            return val

        # compute log reference length
        self.ref_len = np.log(float(len(self.crefs)))

        scores = []

        for test, refs in zip(self.ctest, self.crefs):
            # compute vector for test captions
            vec, norm, length = counts2vec(test)
            # compute vector for ref captions
            score = np.array([0.0 for _ in range(self.n)])
            for ref in refs:
                vec_ref, norm_ref, length_ref = counts2vec(ref, is_ref=True)
                score += sim(vec, vec_ref, norm, norm_ref, length, length_ref)
            # change by vrama91 - mean of ngram scores, instead of sum
            score_avg = np.mean(score)
            # divide by number of references
            score_avg /= len(refs)
            # multiply score by 10 #TODO: Check if times 10 is needed
            #score_avg *= 10.0
            # append score of an image to the score list
            scores.append(score_avg)
        return scores

    def compute_score(self, option=None, verbose=0):
        # compute idf
        # self.compute_doc_freq()
        # assert to check document frequency
        #assert(len(self.ctest) >= max(self.document_frequency.values()))
        # compute cider score
        score = self.compute_cider()
        # debug
        # print score
        return np.mean(np.array(score)), np.array(score)