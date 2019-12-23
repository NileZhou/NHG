import argparse
import json
import logging
import numpy
import random
from time import time

import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

import summarunner_weather.models
import summarunner_weather.utils

logging.basicConfig(level=logging.INFO, format='%(asctime)res [INFO] %(message)res')
parser = argparse.ArgumentParser(description='extractive summary')

# model
parser.add_argument('-save_dir', type=str, default='checkpoints/')
parser.add_argument('-embed_dim', type=int, default=100)
parser.add_argument('-embed_num', type=int, default=100)
parser.add_argument('-pos_dim', type=int, default=50)
parser.add_argument('-pos_num', type=int, default=100)
parser.add_argument('-seg_num', type=int, default=10)
parser.add_argument('-kernel_num', type=int, default=100)
parser.add_argument('-kernel_sizes', type=list, default=[3, 4, 5])
parser.add_argument('-model', type=str, default='RNN_RNN')
parser.add_argument('-hidden_size', type=int, default=200)
# train
parser.add_argument('-lr', type=float, default=1e-4)
parser.add_argument('-batch_size', type=int, default=16)
parser.add_argument('-epochs', type=int, default=30)
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-train_dir', type=str, default='../../data/chinese/cont2sum/little/train.json')
parser.add_argument('-embedding', type=str, default='../../data/chinese/cont2sum/little/embedding.npz')
parser.add_argument('-word2id', type=str, default='../../data/chinese/cont2sum/little/word2id.json')
parser.add_argument('-report_every', type=int, default=1500)
parser.add_argument('-seq_trunc', type=int, default=50)
parser.add_argument('-max_norm', type=float, default=1.0)
# test
parser.add_argument('-load_dir', type=str, default='checkpoints/AttnRNN_seed_1.pt')
parser.add_argument('-test_dir', type=str, default='../../data/chinese/cont2sum/little/test.json')
parser.add_argument('-ref', type=str, default='outputs/ref')
parser.add_argument('-hyp', type=str, default='outputs/hyp')
parser.add_argument('-topk', type=int, default=3)
# device
parser.add_argument('-device', type=int, default=0)
# option
parser.add_argument('-test', action='store_true')
parser.add_argument('-debug', action='store_true')
parser.add_argument('-predict', action='store_true')
args = parser.parse_args()
use_gpu = args.device is not None


# set cuda device and seed
torch.cuda.set_device(args.device)
torch.cuda.manual_seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)
numpy.random.seed(args.seed)
random_state = args.seed


def eval(net, vocab, data_iter, criterion):
    net.eval()
    total_loss = 0
    batch_num = 0
    for batch in data_iter:
        features, targets, _, doc_lens = vocab.make_features(batch)
        features, targets = Variable(features), Variable(targets.float())
        if use_gpu:
            features = features.cuda()
            targets = targets.cuda()
        probs = net(features, doc_lens)
        loss = criterion(probs, targets)
        # origin: total_loss += loss.data[0]
        total_loss += loss.data.item()
        batch_num += 1
    loss = total_loss / batch_num
    net.train()
    return loss


def train(n_val=50):
    """
    验证集条数
    :param n_val:
    :return:
    """
    logging.info('Loading vocab,train and val dataset.Wait a second,please')

    embed = torch.Tensor(np.load(args.embedding)['embedding'])
    with open(args.word2id) as f:
        word2id = json.load(f)
    vocab = summarunner_weather.utils.Vocab(embed, word2id)

    with open(args.train_dir) as f:
        examples = [json.loads(line) for line in f]

    train_dataset = summarunner_weather.utils.Dataset(examples[: -n_val])

    val_dataset = summarunner_weather.utils.Dataset(examples[-n_val:]) # 从train数据集中拿n_val条做验证集

    # update args
    args.embed_num = embed.size(0)
    args.embed_dim = embed.size(1)
    # build model
    net = getattr(summarunner_weather.models, args.model)(args, embed)
    if use_gpu:
        net.cuda()
    # load dataset
    train_iter = DataLoader(dataset=train_dataset,
                            batch_size=args.batch_size,
                            shuffle=True)
    val_iter = DataLoader(dataset=val_dataset,
                          batch_size=args.batch_size,
                          shuffle=False)
    # loss function
    criterion = nn.BCELoss()
    # model info
    print(net)
    # params = sum(p.numel() for p in list(net.parameters())) / 1e6
    # print('#Params: %.1fM' % params)

    min_loss = float('inf')
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    net.train()

    t1 = time()
    for epoch in range(1, args.epochs + 1):
        print("epoch: ", epoch)
        for i, batch in enumerate(train_iter):
            print("batch num: ", i)
            features, targets, _, doc_lens = vocab.make_features(batch)
            features, targets = Variable(features), Variable(targets.float())
            if use_gpu:
                features = features.cuda()
                targets = targets.cuda()
            probs = net(features, doc_lens)
            loss = criterion(probs, targets)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm(net.parameters(), args.max_norm)
            optimizer.step()
            if args.debug:
                print('Batch ID:%d Loss:%f' % (i, loss.data[0]))
                continue
            if i % args.report_every == 0:
                cur_loss = eval(net, vocab, val_iter, criterion)
                if cur_loss < min_loss:
                    min_loss = cur_loss
                    net.save()
                logging.info('Epoch: %2d Min_Val_Loss: %f Cur_Val_Loss: %f'
                             % (epoch, min_loss, cur_loss))
    t2 = time()
    logging.info('Total Cost:%f h' % ((t2 - t1) / 3600))


def train_k_fold(log_path='checkpoints/train_k_fold_RNN_RNN_info.txt'):
    embed = torch.Tensor(np.load(args.embedding)['embedding'])
    with open(args.word2id) as f:
        word2id = json.load(f)
    vocab = summarunner_weather.utils.Vocab(embed, word2id)

    with open(args.train_dir, 'r', encoding='utf-8') as f:
        examples = [json.loads(line) for line in f]

    train_X = [example['content'] for example in examples]
    train_y = [example['labels'] for example in examples]

    args,embed_num = embed.size(0)
    args.embed_dim = embed.size(1)

    infos = []
    cv_ptr = 0
    for train_index, val_index in KFold(n_splits=10, random_state=random_state, shuffle=True).split(train_X, train_y):
        train_data = [{'content': examples[i]['content'], 'labels': examples[i]['labels'],
                       'summary': examples[i]['summary']} for i in train_index]
        val_data = [{'content': examples[i]['content'], 'labels': examples[i]['labels'],
                     'summary': examples[i]['summary']} for i in val_index]

        train_dataset = summarunner_weather.utils.Dataset(train_data)
        val_dataset = summarunner_weather.utils.Dataset(val_data)

        # build model
        net = getattr(summarunner_weather.models, args.model)(args, embed)
        if use_gpu:
            net.cuda()

        # load dataset
        train_iter = DataLoader(dataset=train_dataset,
                                batch_size=args.batch_size,
                                shuffle=True)
        val_iter = DataLoader(dataset=val_dataset,
                              batch_size=args.batch_size,
                              shuffle=False)
        # loss function
        criterion = nn.BCELoss() # Binary Cross Entropy loss
        # model info
        print(net)
        # params = sum(p.numel() for p in list(net.parameters())) / 1e6
        # print('#Params: %.1fM' % params)

        min_loss = float('inf')
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
        net.train()

        t1 = time()
        train_loss, val_loss = [], []
        for epoch in range(1, args.epochs + 1):
            for i, batch in enumerate(train_iter):
                print("epoch: {}, batch num: {}".format(epoch, i))
                features, targets, _, doc_lens = vocab.make_features(batch)
                features, targets = Variable(features), Variable(targets.float())
                if use_gpu:
                    features = features.cuda()
                    targets = targets.cuda()
                probs = net(features, doc_lens)
                loss = criterion(probs, targets)
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm(net.parameters(), args.max_norm)
                optimizer.step()
                train_loss.append(float(loss.data))
                if args.debug:
                    print('Batch ID:%d Loss:%f' % (i, loss.data[0]))
                    continue
                if i % args.report_every == 0:
                    cur_loss = eval(net, vocab, val_iter, criterion)
                    if cur_loss < min_loss:
                        min_loss = cur_loss
                        net.save(cv_ptr)
                    val_loss.append(cur_loss)
                    logging.info('Epoch: %2d Min_Val_Loss: %f Cur_Val_Loss: %f' % (epoch, min_loss, cur_loss))
        t2 = time()
        logging.info('Total Cost:%f h' % ((t2 - t1) / 3600))

        with open(args.test_dir, 'r', encoding='utf-8') as f:
            test_data = [json.loads(line) for line in f]
        test_dataset = summarunner_weather.utils.Dataset(test_data)
        test_iter = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
        test_loss = eval(net, vocab, test_iter, criterion) # 获取测试集上的loss

        infos.append(json.dumps({'fold': cv_ptr, 'train_lozz': train_loss, 'val_lozz': val_loss, 'test_loss': test_loss,
                                 'time_cost': (t2 - t1) / 3600}))

        cv_ptr += 1

    with open(log_path, 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(infos))


def train_3_model_k_fold(model_prefixies, save_dir):
    for model_prefix in model_prefixies:
        args.model = model_prefix
        log_path = os.path.join(save_dir, "train_k_fold_" + model_prefix + "_info.txt")
        train_k_fold(log_path)


if __name__ == '__main__':
    models = ["RNN_RNN", 'CNN_RNN', 'AttnRNN']
    # train_k_fold()
    # train()
    train_3_model_k_fold(models, "checkpoints") # 对3个模型进行k折交叉验证
