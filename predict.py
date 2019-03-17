import argparse
import json
import logging
import numpy
import os
import random
from time import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch.utils.data import DataLoader
from tqdm import tqdm

import summarunner_can_run.models
import summarunner_can_run.utils


def predict(load_dir="checkpoints/RNN_RNN_seed_1.pt", embedding_path="data/embedding.npz", word2id_path="data/word2id.json",
            test_path="tst.txt", essay_trunc=100, sent_trunc=50, top_k=2, use_gpu=True):
    embed = torch.Tensor(np.load(embedding_path)['embedding'])
    with open(word2id_path) as f:
        word2id = json.load(f)
    vocab = summarunner_can_run.utils.Vocab(embed, word2id)
    if use_gpu:
        checkpoint = torch.load(load_dir, map_location='cuda:0')
    else:
        checkpoint = torch.load(load_dir, map_location=lambda storage, loc: storage)
    # checkpoint['args']['device'] saves the device used as train time
    # if at test time, we are using a CPU, we must override device to None
    if not use_gpu:
        checkpoint['args'].device = None
    net = getattr(summarunner_can_run.models, checkpoint['args'].model)(checkpoint['args'])
    net.load_state_dict(checkpoint['model'])
    if use_gpu:
        net.cuda()
    net.eval()
    # make features
    with open(test_path, 'r') as f:
        essay = ''.join(f.readlines())
    sents = essay.split('\n')
    max_sent_num = min(essay_trunc, len(sents)) # 最多允许essay_trunc个句子
    sents = sents[:max_sent_num] # 这是新的句子list
    max_sent_len = 0
    batch_sents = []
    for sent in sents:
        words = sent.split()
        words = words[:min(sent_trunc, len(words))]
        max_sent_len = len(words) if len(words) > max_sent_len else max_sent_len
        batch_sents.append(words)
    features = []
    for sent in batch_sents:
        feature = [vocab.w2i(w) for w in sent] + [vocab.PAD_IDX for _ in range(max_sent_len - len(sent))]
        features.append(feature)
    features = torch.LongTensor(features)
    if use_gpu:
        probs = net(Variable(features).cuda(), [len(sents)])
    else:
        probs = net(Variable(features), [len(sents)])

    topk_indices = probs.topk(top_k)[1].cpu().data.numpy()
    topk_indices.sort()
    hyp = [sents[index] for index in topk_indices]
    print(''.join(hyp))


if __name__ == "__main__":
    predict()




