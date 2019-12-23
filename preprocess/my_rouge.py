import rouge
import json
import numpy as np
from numpy.linalg import norm

# we use the rouge package: https://github.com/Diego999/py-rouge
# pip install py-rouge

# content: seq1[w1, w2, w3], seq2, seq3
# title: seq4[w4, w5]
# for i in seq1:
#    a = compute_sim(w_i, w4)
#    b = compute_sim(w_i, w5)
#    score_i = max(a, b)
# scores: list{score_ 1 --> 3}
# new_rouge_1(seq1, title) = avg(scores)


# new_rouge_2: 把content , title 的2-gram重复以上步骤


class Myrouge:
    """
    论文中的计算方式,来源于: 《Learning to Extract Coherent Summary via Deep Reinforcement Learning》
    """
    def __init__(self, w1=0.4, w2=1.0, wl=0.5, alpha=0.5, weight_factor=1.2):
        metrics = ['rouge-n', 'rouge-l']
        self.evaluator = rouge.Rouge(metrics=metrics, max_n=2, alpha=alpha)
        self.w1 = w1
        self.w2 = w2
        self.wl = wl

    def compute(self, hyp, ref):
        scores = self.evaluator.get_scores(hyp, ref)
        rouge1 = scores['rouge-1']['r']
        rouge2 = scores['rouge-2']['r']
        rougel = scores['rouge-l']['r']
        return self.w1 * rouge1 + self.w2 * rouge2 + self.wl * rougel


