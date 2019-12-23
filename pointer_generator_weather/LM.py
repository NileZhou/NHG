import math
import pickle
import codecs
import ujson as json
import re
import os

# 11019603条 语料数据


class WeatherTriGram:
    def __init__(self, corpus_path='/media/nile/study/repositorys/autosumma/data/chinese/lan/language_model_corpus.txt',
                 weather_vocab_path='../data/chinese/lan/weather_vocab.json', step=3, lam=1):
        """
        定义在sohu词表但不在weather词表中的词为超出词(oow)
        每个oov统一变为O(out).
        基于N-gram的统计语言模型
        建立起本地word-count库
        建立起本地词表库
        :param corpus_path: 训练数据集路径
        :param weather_vocab_path: 气象词频统计表大小
        :param step: N元模型的N
        :param lam:
        """
        self.non_chinese_pat = re.compile(r'[a-zA-Z0-9０１２３４５６７８９，、。？“”！：；‘’《》%…·]+')
        self.step = step
        self.lam = lam

        # 先读入weather_vocab
        with open(weather_vocab_path, 'r', encoding='utf-8') as f:
            weather_vocab = json.load(f)
        tmp_words = list(weather_vocab.keys())
        self.weather_words = set([word for word in tmp_words if re.search(self.non_chinese_pat, word) is None])
        print('原始气象词表(去除各种英文数字符号后的词表)大小: ', len(self.weather_words)) # 7695

        # 开始做sohu数据集的一元词频统计(一元词频统计)
        # 与此同时，开始做 word -> id 的映射词典
        if os.path.exists('../data/chinese/lan/1-gram.json'):
            with open('../data/chinese/lan/1-gram.json', 'r', encoding='utf-8') as f:
                self.freq1 = json.load(f)
            # TODO 读取映射词典

        else:
            self.word2id = dict()
            self.freq1 = dict()
            # <start>与<end>是为2-gram做准备
            self.freq1['<start>'] = 0
            self.word2id['<start>'] = 0
            self.freq1['<end>'] = 0
            self.word2id['<end>'] = 1
            id = 2 # id现在从2开始算起了
            with open(corpus_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f.readlines()):
                    if i > 0 and i % 10000 == 0:
                        print('{} 1-gram has processed.'.format(i))
                    words = line.strip().split()
                    # 首先过滤掉没必要统计的词
                    tokens = [token for token in words if re.search(self.non_chinese_pat, token) is None]
                    if len(tokens) < 3: continue # 3个词都没有的句子，不在考虑范围内(因为生成标题长度至少为3)
                    for token in tokens:
                        if token in self.weather_words:  # 在气象数据集词表中，再进行统计
                            if token not in self.freq1:
                                self.freq1[token] = 1
                                self.word2id[token] = id
                                id += 1
                            else:
                                self.freq1[token] += 1
                    self.freq1['<start>'] += 1
                    self.freq1['<end>'] += 1
            with open('../data/chinese/lan/1-gram.json', 'w', encoding='utf-8') as f:
                json.dump(self.freq1, f, ensure_ascii=False)
            with open('../data/chinese/lan/word2id.json', 'w', encoding='utf-8') as f:
                json.dump(self.word2id, f, ensure_ascii=False)

        print('num of 1-gram: ', len(self.freq1))

        # 开始做sohu数据集的二元词频统计
        if os.path.exists('../data/chinese/lan/2-gram.json'):
            with open('../data/chinese/lan/2-gram.json', 'r', encoding='utf-8') as f:
                self.freq2 = json.load(f)
        else:
            self.freq2 = dict() # word1-word2 : frequent
            # 每句话前面要加上<start> , 后面加上<end>
            with open(corpus_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f.readlines()):
                    if i > 0 and i % 10000 == 0:
                        print('{} 2-gram has processed.'.format(i))
                    words = line.strip().split()
                    tokens = [token for token in words if re.search(self.non_chinese_pat, token) is None]
                    if len(tokens) < 3: continue # 3个词都没有的句子，不在考虑范围内(因为生成标题长度至少为3)
                    tokens.insert(0, '<start>')
                    tokens.insert(len(tokens), '<end>')
                    for j in range(len(tokens) - 1):
                        word1, word2 = tokens[j], tokens[j+1]
                        if word1 not in self.freq1 or word2 not in self.freq1:
                            continue
                        pseudo_word = word1 + '-' + word2
                        if pseudo_word not in self.freq2:
                            self.freq2[pseudo_word] = 1
                        else:
                            self.freq2[pseudo_word] += 1
            with open('../data/chinese/lan/2-gram.json', 'w', encoding='utf-8') as f:
                json.dump(self.freq2, f, ensure_ascii=False)
        print('num of 2-gram: ', len(self.freq2))

        # 开始做sohu数据集的三元词频统计
        if os.path.exists('../data/chinese/lan/3-gram.json'):
            with open('../data/chinese/lan/3-gram.json', 'r', encoding='utf-8') as f:
                self.freq3 = json.load(f)
        else:
            self.freq3 = dict()
            # 2-gram与3-gram起始与终止的词频统计相同
            for k in self.freq2.keys():
                if '<start>' in k:
                    pseudo_word = '<start>-' + k
                    self.freq3[pseudo_word] = self.freq2[k]
                elif '<end>' in k:
                    pseudo_word = k + '-<end>'
                    self.freq3[pseudo_word] = self.freq2[k]
            with open(corpus_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f.readlines()):
                    if i > 0 and i % 10000 == 0:
                        print('{} 3-gram has processed.'.format(i))
                    words = line.strip().split()
                    tokens = [token for token in words if re.search(self.non_chinese_pat, token) is None]
                    if len(tokens) < 3: continue  # 3个词都没有的句子，不在考虑范围内(因为生成标题长度至少为3)
                    tokens.insert(0, '<start>')
                    tokens.insert(len(tokens), '<end>')
                    for j in range(len(tokens)-2):
                        word1, word2, word3 = tokens[j], tokens[j+1], tokens[j+2]
                        if word1 not in self.freq1 or word2 not in self.freq1 or word3 not in self.freq1:
                            continue
                        pseudo_word = word1 + '-' + word2 + '-' + word3
                        if pseudo_word not in self.freq3:
                            self.freq3[pseudo_word] = 1
                        else:
                            self.freq3[pseudo_word] += 1
            with open('../data/chinese/lan/3-gram.json', 'w', encoding='utf-8') as f:
                json.dump(self.freq3, f, ensure_ascii=False)
        print('num of 3-gram: ', len(self.freq3))

    def compute_sentence_prob(self, sentence):
        words = sentence.split()
        if len(words) < 3:
            # print('the num of words of sentence is smaller than 3.')
            return -100
        else:
            words.insert(0, '<start>')
            words.insert(0, '<start>')
            words.insert(len(words), '<end>')
            words.insert(len(words), '<end>')
            # 先算起始词的概率
            start_2_gram = '<start>-'+words[0]
            if start_2_gram in self.freq2:
                prob = math.log((self.freq2[start_2_gram] + 1) / (self.freq1['<start>'] + len(self.freq1)))
            else:
                prob = math.log(1 / (self.freq1['<start>'] + len(self.freq1)))
            # eg: sentence: 中国气象局 刘雅鸣 出席 会议 服务 工作
            # psedudo_word:
            # 前面的prob: <start>-<start>-中国气象局
            #     <start>-中国气象局-刘雅鸣
            #     中国气象局-刘雅鸣-出席
            #     刘雅鸣-出席-会议
            #     出席-会议-服务
            #     会议-服务-工作
            #     服务-工作-<end>
            #     工作-<end>-<end>
            cnt = 0
            for i in range(1, len(words) - 2):
                pseudo_word = words[i]+'-'+words[i+1]+'-'+words[i+2]
                pre_word = words[i]+'-'+words[i+1]
                if pseudo_word in self.freq3:
                    cnt += 1
                    # 前面2个必在self.freq2
                    prob += math.log((self.freq3[pseudo_word] + 1) / (self.freq2[pre_word] + len(self.freq2)))
            if cnt == 0:
                # print('divided by zero!')
                return -2**31
            prob /= cnt

            return prob


if __name__ == '__main__':
    os.chdir('pointer_generator_weather')
    language_model = WeatherTriGram(corpus_path='/media/nile/study/repositorys/autosumma/data/chinese/lan/language_'
                                                'model_corpus.txt')
    res = language_model.compute_sentence_prob('中国 企业 似乎 寻找到 全新 的 路径')
    print(res)
    print(language_model.compute_sentence_prob('中国 企业 似乎 寻找到 它 方法 去 似乎 '))
