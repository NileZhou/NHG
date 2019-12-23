import numpy as np
from numpy.linalg import norm
from functools import lru_cache


class SimRouge:
    """
    利用word embedding中词之间的余弦相似度计算距离
    """
    def __init__(self, word2id, embed, w1=0.4, w2=1.0, wl=0.5):
        self.word2id = word2id
        self.embed = embed
        self.w1 = w1
        self.w2 = w2
        self.wl = wl

    def _word_sim(self, word1, word2):
        """
        遇到OOV怎么办
        """
        if isinstance(word1, str):
            vec1 = self.embed[self.word2id[word1]]
            vec2 = self.embed[self.word2id[word2]]
            return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))
        else:
            return np.dot(word1, word2) / (norm(word1) * norm(word2))

    @lru_cache()
    def compute_sim_n(self, hyp, ref, n=2):
        words1 = hyp.split()
        words2 = ref.split()
        len1, len2 = len(words1), len(words2)
        assert len1 >= n and len2 >= n
        scores = []
        for i in range(len1 - n):
            imax = -1
            grams1 = words1[i: i + n]
            vec1 = np.mean([self.embed[self.word2id[w]] for w in grams1], axis=0, dtype=np.float32)
            if np.mean(vec1) == 0:
                scores.append(0)
                continue
            if len2 == n:
                grams2 = words2
                vec2 = np.mean([self.embed[self.word2id[w]] for w in grams2], axis=0, dtype=np.float32)
                if np.mean(vec2) == 0:
                    imax = 0
                else:
                    imax = self._word_sim(vec1, vec2)
            for j in range(len2 - n):
                grams2 = words2[j: j+n]
                # 累加之后计算平均值
                vec2 = np.mean([self.embed[self.word2id[w]] for w in grams2], axis=0, dtype=np.float32)
                if np.mean(vec2) == 0:
                    score = 0
                else:
                    score = self._word_sim(vec1, vec2)
                imax = max(imax, score)
            scores.append(imax) # 这一个n-gram与title整句话的2n-gram的最大相似度
        if len1 == n:
            vec1 = np.mean([self.embed[self.word2id[w]] for w in words1], axis=0, dtype=np.float32)
            if np.mean(vec1) == 0:
                return 0
            else:
                if len2 == n:
                    vec2 = np.mean([self.embed[self.word2id[w]] for w in words2], axis=0, dtype=np.float32)
                    if np.mean(vec2) == 0:
                        return 0
                    else:
                        return self._word_sim(vec1, vec2)
                else:
                    imax = -1
                    for j in range(len2 - n):
                        grams2 = words2[j: j + n]
                        # 累加之后计算平均值
                        vec2 = np.mean([self.embed[self.word2id[w]] for w in grams2], axis=0, dtype=np.float32)
                        if np.mean(vec2) == 0:
                            score = 0
                        else:
                            score = self._word_sim(vec1, vec2)
                        imax = max(imax, score)
                    scores.append(imax)

        score = np.mean(scores, dtype=np.float32)
        # scores = sorted(scores, reverse=True)
        # score = np.mean(scores[: int(1/3*len(scores))], dtype=np.float32)

        return score

    def compute_sim_l(self, hyp, ref):
        min_len = min(len(hyp.split()), len(ref.split()))
        scores = []
        for n in range(1, min_len+1):
            scores.append(self.compute_sim_n(hyp, ref, n))
        max_len = np.argmax(scores) + 1
        return max_len / len(ref)

    def compute(self, hyp, ref, replace_UNK=False):
        if replace_UNK:            # 对title 进行改造(如果word2id里没有，判为<UNK TOKEN>)
            new_words = []
            words = ref.split()
            for word in words:
                if self.word2id.get(word, None) is None:
                    new_words.append('UNK_TOKEN')
                else:
                    new_words.append(word)
            ref = ' '.join(new_words)
        min_len = min(len(hyp.split()), len(ref.split()))
        if min_len < 2:
            return self.w1 * self.compute_sim_n(hyp, ref, 1)
        sim1 = self.compute_sim_n(hyp, ref, 1)
        sim2 = self.compute_sim_n(hyp, ref, 2)
        siml = self.compute_sim_l(hyp, ref)
        return self.w1 * sim1 + self.w2 * sim2 + self.wl * siml




# with open('/media/nile/study/repositorys/autosumma/summarunner_weather/utils/word2id.json') as f:
#     word2id = json.load(f)
#
# word2id = dict([(v, k) for k, v in word2id.items()])
#
# embed = np.load('/media/nile/study/repositorys/autosumma/summarunner_weather/utils/embedding.npz')['embedding']
# sim_rouge = SimRouge(word2id, embed)
