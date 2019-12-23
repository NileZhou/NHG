import json

import numpy as np
import torch
from torch.autograd import Variable

import summarunner_weather.models
import summarunner_weather.utils


def summa_weather_predict(essay, model_path, embedding_path, word2id_path, essay_trunc, sent_trunc, top_k, use_gpu):
    embed = torch.Tensor(np.load(embedding_path)['embedding'])
    with open(word2id_path) as f:
        word2id = json.load(f)
    vocab = summarunner_weather.utils.Vocab(embed, word2id)
    if use_gpu:
        checkpoint = torch.load(model_path + '_seed_1.pt', map_location='cuda:0')
    else:
        checkpoint = torch.load(model_path + '_seed_1.pt', map_location=lambda storage, loc: storage)
        checkpoint['args'].device = None

    net = getattr(summarunner_weather.models, checkpoint['args'].model)(checkpoint['args'])
    net.load_state_dict(checkpoint['model'])
    if use_gpu: net.cuda()
    net.eval()
    sents = essay.split('\n')
    max_sent_num = min(essay_trunc, len(sents))
    sents = sents[:max_sent_num]
    max_sent_len = 0
    batch_sents = []
    for sent in sents:
        words = sent.split()
        words = words[: min(sent_trunc, len(words))]
        max_sent_len = len(words) if len(words) > max_sent_len else max_sent_len
        batch_sents.append(words)
    features = []
    for sent in batch_sents:
        feature = [vocab.w2i(w) for w in sent] + [vocab.PAD_IDX for _ in range(max_sent_len - len(sent))]
        features.append(feature)
    features = torch.LongTensor(features)
    if use_gpu:
        probs = net(Variable(features).cuda(), [len(sents)], use_gpu)
    else:
        probs = net(Variable(features), [len(sents)], use_gpu)

    topk_indices = probs.topk(top_k)[1].cpu().data.numpy()
    topk_indices.sort()
    hyp = [''.join(sents[index].split()) for index in topk_indices]
    return '\n'.join(hyp)


def predict(out_path, load_dir="little/checkpoints/RNN_RNN_fold1_seed_1.pt", embedding_path="../data/chinese/cont2sum/little/embedding.npz",
            word2id_path="../data/chinese/cont2sum/little/word2id.json",
            test_path="tst.txt", essay_trunc=100, sent_trunc=50, top_k=2, use_gpu=True):
    embed = torch.Tensor(np.load(embedding_path)['embedding'])
    with open(word2id_path) as f:
        word2id = json.load(f)
    vocab = summarunner_weather.utils.Vocab(embed, word2id)
    if use_gpu:
        checkpoint = torch.load(load_dir, map_location='cuda:0')
    else:
        checkpoint = torch.load(load_dir, map_location=lambda storage, loc: storage)
    # checkpoint['args']['device'] saves the device used as train time
    # if at test time, we are using a CPU, we must override device to None
    if not use_gpu:
        checkpoint['args'].device = None
    net = getattr(summarunner_weather.models, checkpoint['args'].model)(checkpoint['args'])
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
    hyp = [''.join(sents[index].split()) for index in topk_indices]

    with open(out_path, 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(hyp))
    print('\n'.join(hyp))


if __name__ == "__main__":
    out_path = 'output.txt'
    print('抽取摘要: \n')
    predict(out_path)
    print('\n\n\n抽取完毕，请检查； %s' % out_path)



