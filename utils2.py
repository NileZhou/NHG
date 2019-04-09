import os
import re
import json
import rouge
import subprocess
from multiprocessing.dummy import Pool
import numpy as np
from typing import NamedTuple, List, Callable, Dict, Tuple, Optional
from collections import Counter
from random import shuffle
from functools import lru_cache
import torch

word_detector = re.compile('\w')  # 只要是字母，数字，下划线或星号都算词语(因为这里要把括号，引号, 中间间隔的横线这些排除掉)
non_word_char_in_word = re.compile(r'(?<=\w)\W(?=\w)')  # 这样感觉其实不好，处理方式太过于粗暴


class Vocab:
    PAD = 0  # zero padding
    SOS = 1  # start of sentence
    EOS = 2  # end of sentence
    UNK = 3  # unknown token

    def __init__(self):
        self.word2index = {}
        self.word2count = Counter()
        self.reserved = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
        self.index2word = self.reserved[:]
        self.embeddings = None

    def __getitem__(self, item):
        """
        获取index为item对应的词, 或获取词对应的index(找不到返回3(UNK))
        :param item:
        :return:
        """
        if type(item) is int:
            return self.index2word[item]
        return self.word2index.get(item, self.UNK)  # 没这个词，直接返回UNK

    def __len__(self):
        return len(self.index2word)

    def add_words(self, words: List[str]):
        for word in words:
            if word not in self.word2index:
                self.word2index[word] = len(self.index2word)
                self.index2word.append(word)
        self.word2count.update(words)  # 直接输进去一堆，Count都能进行计数，牛逼

    def trim(self, vocab_size: int, min_freq: int = 1):
        """
        根据要求的词汇表大小对现有词表进行修剪，并且硬性限制最小的词语出现频数
        :param vocab_size:
        :param min_freq:
        :return:
        """
        if len(self.index2word) <= vocab_size:
            print('the size of vocabulary is {}, it\'s smaller than vocabsize: {}'.format(len(self.index2word),
                                                                                          vocab_size))
            return
        ordered_words = sorted([(c, w) for (w, c) in self.word2count.items()], reverse=True)
        ordered_words = ordered_words[: vocab_size]
        self.word2index = {}
        self.word2count = Counter()
        self.index2word = self.reserved[:]
        for count, word in ordered_words:
            if count >= min_freq:
                self.word2index[word] = len(self.index2word)
                self.word2count[word] = count
                self.index2word.append(word)

    def load_embeddings(self, embed_path: str, dtype=np.float32) -> int:
        """
        把存在于当前Vocabulary中的词，又没有对应embedding的进行embedding填充
        :param embed_path:
        :param dtype:
        :return:
        """
        load_num = 0
        vocab_size = len(self)
        with open(embed_path, 'r') as f:
            for line in f.readlines():
                data = line.strip().split()
                word = data[0]
                data.pop(0)
                idx = self.word2index.get(word, None)  # word -> id
                if idx is not None:
                    vec = np.array(data, dtype=dtype)
                    if self.embeddings is None:  # 如果当前没有embedding,按照传入vec的shape进行构造随机矩阵
                        n_dims = len(vec)
                        self.embeddings = np.random.normal(np.zeros((vocab_size, n_dims))).astype(dtype)
                        self.embeddings[self.PAD] = np.zeros(n_dims)
                    self.embeddings[idx] = vec
                    load_num += 1

        return load_num

    @lru_cache(maxsize=None)
    def is_word(self, token_id: int) -> bool:
        """
        Least Recent Used algorithm. 判断一个id是否是一个单词
        """
        if token_id < 4: return False
        if token_id >= len(self): return False
        token_str = self.index2word[token_id]
        if not word_detector.search(token_str) or token_str == '<P>':
            return False
        return True


class OOVDict:
    """
    记录超出词词典
    """

    def __init__(self, base_oov_idx):
        """
        从这里可以看出，这样写注释确实有极大的好处
        :param base_oov_idx:
        """
        self.word2index = {}  # Dict[Tuple[int, str], int] Dict[(batch中第i句话, 单词), word对应索引]
        self.index2word = {}  # Dict[Tuple[int, int], str]
        self.next_index = {}  # Dict[int, int]
        self.base_oov_idx = base_oov_idx  # base_oov_idx存储的是基本的词典大小
        self.ext_vocab_size = base_oov_idx  # 扩展后的词表大小

    def add_word(self, idx_in_batch: int, word) -> int:
        """
        在词汇表中添加一个词
        :param idx_in_batch:
        :param word:
        :return:
        """
        key = (idx_in_batch, word)
        index = self.word2index.get(key)
        if index is not None: return index
        index = self.next_index.get(idx_in_batch, self.base_oov_idx)
        self.next_index[idx_in_batch] = index + 1
        self.word2index[key] = index
        self.index2word[(idx_in_batch, index)] = word
        self.ext_vocab_size = max(self.ext_vocab_size, index + 1)
        return index


# 一条训练数据
class Example(NamedTuple):
    src: List[str]
    tgt: List[str]
    src_len: int
    tgt_len: int


# 一块训练数据
class Batch(NamedTuple):
    examples: List[Example]
    input_tensor: Optional[torch.Tensor]
    target_tensor: Optional[torch.Tensor]
    src_lens: List[int]
    oov_dict: Optional[OOVDict]

    @property
    def ext_vocab_size(self):
        if self.oov_dict is not None:
            return self.oov_dict.ext_vocab_size
        return None


def text_tokenizer(text: str, newline: str = None):
    """
    支持普通切分或把\n替换为新的字符后进行切分
    :param text:
    :param newline:
    :return:
    """
    if newline is not None:
        text = text.replace('\n', ' ' + newline + ' ')
    return text.split()


class Dataset:
    def __init__(self, filename: str, tokenize: Callable = text_tokenizer, max_src_len=60,
                 max_tgt_len=15, truncate_src=True, truncate_tgt=True):
        """
        :param filename: ~/Downloads/sum2tit.txt
        :param tokenize: 分词器
        :param max_src_len: X 最大词数
        :param max_tgt_len: Y 最大词数
        :param truncate_src: 是否对src做truncate
        :param truncate_tgt: 是否对tgt做truncate
        """
        print('Reading dataset %s ...' % filename)
        self.filename = filename
        self.pairs = []
        self.src_len = 0  # 记录X最大词数
        self.tgt_len = 0  # 记录Y最大词数
        with open(filename, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f.readlines()):
                tmp = json.loads(line)
                cols = list(tmp.keys())  # 获取json的列名
                col1, col2 = cols[0], cols[1]  # 按照字典序排序
                src = tokenize(tmp[col1])
                if max_src_len and len(src) > max_src_len:
                    if truncate_src:
                        src = src[:max_src_len]
                    else:
                        continue
                tgt = tokenize(tmp[col2])
                if max_tgt_len and len(tgt) > max_tgt_len:
                    if truncate_tgt:
                        tgt = tgt[:max_tgt_len]
                    else:
                        continue
                src_len = len(src) + 1  # EOS
                tgt_len = len(tgt) + 1  # EOS
                self.src_len = max(self.src_len, src_len)
                self.tgt_len = max(self.tgt_len, tgt_len)
                self.pairs.append(Example(src, tgt, src_len, tgt_len))
        print('%d pairs.' % len(self.pairs))

    def build_vocab(self, max_vocab_size, embed_file_path) -> Vocab:
        """
        建立七词表(并加载embedding)
        数据集名如果是XXX.txt
        词汇名得是   XXX.{词表大小}.vocab
        :param max_vocab_size:
        :param embed_file_path:
        :return:
        """
        filename, _ = os.path.splitext(self.filename)  # splitext()会将最后一个点及其之前作为第一部分，这个点之后作为第二部分
        if max_vocab_size:
            filename += '.%d' % max_vocab_size
        filename += '.vocab'
        if os.path.isfile(filename):
            vocab = torch.load(filename)
            print('loading vocabulary which exist %d words here.' % len(vocab))
        else:
            print('building vocabulary...')
            vocab = Vocab()
            for example in self.pairs:
                vocab.add_words(example.src)
                vocab.add_words(example.tgt)
            vocab.trim(vocab_size=max_vocab_size)  # 最后要修剪下词表大小
            print('now we have a vocabulary which exist %d words' % len(vocab))
            torch.save(vocab, filename)
        if embed_file_path:
            load_num = vocab.load_embeddings(embed_file_path)
            print('%d pretrained embeddings loaded.' % load_num)
        return vocab

    def generator(self, batch_size, vocab, ext_vocab=False):
        """
        按batch size生成数据集
        :param batch_size:
        :param src_vocab:
        :param tgt_vocab:
        :param ext_vocab: extension vocab.对每个batch中文章中出现的新词做一个新的词表. pointer generator使用时就要把这里改成True
        :return:
        """
        ptr = len(self.pairs)  # 从名称可以推断这是一个指针
        if ext_vocab:
            # 如果档案有词表存在, 让base_oov_idx为已知词表长度
            assert vocab is not None
            base_oov_idx = len(vocab)
        while True:
            if ptr + batch_size > len(self.pairs):
                shuffle(self.pairs)
                ptr = 0
            examples = self.pairs[ptr: ptr + batch_size]
            ptr += batch_size
            src_tensor, tgt_tensor = None, None
            src_lens, oov_dict = None, None
            if vocab:
                examples.sort(key=lambda x: -x.src_len)  # 先按文章长度由长到短排序
                src_lens = [x.src_len for x in examples]
                max_src_len = src_lens[0]
                src_tensor = torch.zeros(max_src_len, batch_size, dtype=torch.long)
                if ext_vocab:
                    oov_dict = OOVDict(base_oov_idx)
                max_tgt_len = max([x.tgt_len for x in examples])
                tgt_tensor = torch.zeros(max_tgt_len, batch_size, dtype=torch.long)
                for i, example in enumerate(examples):
                    for j, word in enumerate(example.src):
                        idx = vocab[word]
                        if ext_vocab and idx == vocab.UNK:
                            idx = oov_dict.add_word(i, word)
                        src_tensor[j, i] = idx
                    src_tensor[example.src_len - 1, i] = vocab.EOS
                    for j, word in enumerate(example.tgt):
                        idx = vocab[word]
                        if ext_vocab and idx == vocab.UNK:
                            idx = oov_dict.word2index.get((i, word), idx)
                        tgt_tensor[j, i] = idx
                    tgt_tensor[example.tgt_len - 1, i] = vocab.EOS

            yield Batch(examples, src_tensor, tgt_tensor, src_lens, oov_dict)


class Hypothesis:
    def __init__(self, tokens, log_probs, dec_hidden, dec_states, enc_attn_weights, num_non_words):
        """

        :param tokens: List[int]
        :param log_probs: List[Float]
        :param dec_hidden: shape: (1, 1, hidden_size)
        :param dec_states: List[dec_hidden]
        :param enc_attn_weights: List[shape: (1, 1, src_len)]
        :param num_non_words: int
        """
        self.tokens = tokens
        self.log_probs = log_probs
        self.dec_hidden = dec_hidden
        self.dec_states = dec_states
        self.enc_attn_weights = enc_attn_weights
        self.num_non_words = num_non_words

    def __repr__(self):
        # 通过repr,不仅能够输出字符串，还能输出字符串的类型信息
        return repr(self.tokens)

    def __len__(self):
        # 只计算word的长度(token不一定是长度)
        return len(self.tokens) - self.num_non_words

    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.log_probs)

    def create_next(self, token, log_prob, dec_hidden, add_dec_states, enc_attn, non_word):
        return Hypothesis(tokens=self.tokens + [token], log_probs=self.log_probs + [log_prob], dec_hidden=dec_hidden,
                          dec_states=self.dec_states + [dec_hidden] if add_dec_states else self.dec_states,
                          enc_attn_weights=self.enc_attn_weights + [enc_attn]
                          if enc_attn is not None else self.enc_attn_weights,
                          num_non_words=self.num_non_words + 1 if non_word else self.num_non_words)


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

        rouge1 = scores['rouge-1']['f'] # 原来是r,改用f
        rouge2 = scores['rouge-2']['f']
        rougel = scores['rouge-l']['f']
        return self.w1 * rouge1 + self.w2 * rouge2 + self.wl * rougel

    def compute_rouges(self, hyps, refs):
        res = []
        for hyp, ref in zip(hyps, refs):
            hyp = ' '.join(hyp)
            ref = ' '.join(ref)
            res.append(self.compute(hyp, ref))
        return res
