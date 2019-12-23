#!/usr/bin/env python3

import json
import numpy as np
from collections import OrderedDict
from glob import glob
from time import time
from multiprocessing import Pool, cpu_count
from itertools import chain


# 从embedding中建立word2id和新的embedding
def build_vocab(embed_path="data/100.w2v", vocab_path="data/embedding.npz", word2id_path="data/word2id.json"):
    print('start building vocab')

    PAD_IDX = 0
    UNK_IDX = 1
    PAD_TOKEN = 'PAD_TOKEN'
    UNK_TOKEN = 'UNK_TOKEN'
    
    f = open(embed_path) # 打开嵌入文件,如100.w2v
    embed_dim = int(next(f).split()[1])

    word2id = OrderedDict()

    # 把PAD_TOKEN和UNK_TOKEN换成此程序用的
    word2id[PAD_TOKEN] = PAD_IDX
    word2id[UNK_TOKEN] = UNK_IDX
    
    embed_list = [] # 一个大矩阵
    # fill PAD and UNK vector
    embed_list.append([0 for _ in range(embed_dim)])
    embed_list.append([0 for _ in range(embed_dim)])
    
    # build Vocab
    for line in f:
        tokens = line.split()
        word = tokens[:-1*embed_dim][0] # 取词
        vector = [float(num) for num in tokens[-1*embed_dim:]] # 后面的小数向量
        embed_list.append(vector)
        word2id[word] = len(word2id)
    f.close()
    embed = np.array(embed_list, dtype=np.float32)
    np.savez_compressed(file=vocab_path, embedding=embed) # npz文件存储的是处理后的embedding
    with open(word2id_path, 'w') as f:
        json.dump(word2id, f)


def worker(files):
    examples = []
    for f in files:
        parts = open(f, encoding='latin-1').read().split('\n\n')
        try:
            entities = {line.strip().split(':')[0]: line.strip().split(':')[1].lower() for line in parts[-1].split('\n')}
        except:
            continue
        sents, labels, summaries = [], [], []
        # content
        for line in parts[1].strip().split('\n'):
            content, label = line.split('\t\t\t')
            tokens = content.strip().split()
            for i, token in enumerate(tokens):
                if token in entities:
                    tokens[i] = entities[token]
            label = '1' if label == '1' else '0'
            sents.append(' '.join(tokens))
            labels.append(label)
        # summary
        for line in parts[2].strip().split('\n'):
            tokens = line.strip().split()
            for i, token in enumerate(tokens):
                if token in entities:
                    tokens[i] = entities[token]
            line = ' '.join(tokens).replace('*', '')
            summaries.append(line)
        ex = {'doc': '\n'.join(sents), 'labels': '\n'.join(labels), 'summaries': '\n'.join(summaries)}
        examples.append(ex)
    return examples


def build_dataset(worker_num=1, source_dir="data/neuralsum/dailymail/validation/*", target_dir="data/val.json"):
    t1 = time()
    
    print('start building dataset')
    if worker_num == 1 and cpu_count() > 1:
        print('[INFO] There are %d CPUs in your device, please increase -worker_num to speed up' % (cpu_count()))
        print("       It'res an IO intensive application, so 2~10 may be a good choice")

    files = glob(source_dir)
    data_num = len(files)
    group_size = data_num // worker_num
    groups = []
    for i in range(worker_num):
        if i == worker_num - 1:
            groups.append(files[i*group_size:])
        else:
            groups.append(files[i*group_size: (i+1)*group_size])
    p = Pool(processes=worker_num)
    multi_res = [p.apply_async(worker, (fs,)) for fs in groups]
    res = [res.get() for res in multi_res]
    
    with open(target_dir, 'w') as f:
        for row in chain(*res):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    t2 = time()
    print('Time Cost : %.1f seconds' % (t2 - t1))


if __name__ == '__main__':
    # build_vocab()
    # or
    # build_dataset()
    pass
