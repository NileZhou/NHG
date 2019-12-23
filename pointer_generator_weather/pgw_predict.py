import torch
import json
import pandas as pd
from pointer_generator_weather.params import Params
from pointer_generator_weather.utils import OOVDict, format_tokens, Dataset
from pointer_generator_weather.test import decode_batch_output
from pointer_generator_weather.model import Seq2Seq


class PGWPredictor:
    def __init__(self, use_gpu=True):
        self.use_gpu=use_gpu
        if use_gpu:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device('cpu')

        p = Params()
        train_status = torch.load(p.model_path_prefix + "_train.pt")
        # print(train_status)
        model_filename = '%s_%02d.pt' % (p.model_path_prefix, train_status['best_epoch_so_far'])  # 取Valid最好的那一轮
        # model = torch.load(model_filename)
        dataset = Dataset(p.data_path, max_src_len=p.max_src_len, max_tgt_len=p.max_tgt_len,
                          truncate_src=p.truncate_src, truncate_tgt=p.truncate_tgt)
        v = dataset.build_vocab(p.vocab_size, embed_file=p.embed_file)
        self.p = p
        self.model = Seq2Seq(v, p, use_gpu=use_gpu)
        self.model.load_state_dict(torch.load(model_filename))
        self.model.to(device)
        # 卧草TMD，原来必须要用eval()固定住结果，不然每次预测的结果都会不一样!
        self.model.eval()
        self.model.encoder.gru.flatten_parameters()
        self.model.decoder.gru.flatten_parameters()

        self.vocab = self.model.vocab
        self.base_oov_idx = len(self.vocab)

    def pgw_predict(self, summary):
        src = summary.split()
        src_len = len(src) + 1  # 最后加一个EOS

        src_tensor = torch.zeros(src_len, 1, dtype=torch.long)
        if self.use_gpu and torch.cuda.is_available():
            src_tensor = src_tensor.cuda()  # 迁移到cuda上去之后(这个函数不是in-place的!!!不像模型)
        oov_dict = OOVDict(self.base_oov_idx)
        for i, word in enumerate(src):
            idx = self.vocab[word]
            if idx == self.vocab.UNK:
                idx = oov_dict.add_word(i, word)
            src_tensor[i, 0] = idx
        src_tensor[src_len - 1, 0] = self.vocab.EOS

        hypotheses = self.model.beam_search(src_tensor, None, oov_dict.ext_vocab_size, 4, min_out_len=self.p.min_out_len,
                                       max_out_len=10, len_in_words=self.p.out_len_in_words)

        to_decode = [hypotheses[0].tokens]
        decoded_batch = decode_batch_output(to_decode, self.vocab, oov_dict)

        return format_tokens(decoded_batch[0])


def pgw_predict_one(summary, use_gpu=True):
    if use_gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device('cpu')

    p = Params()
    train_status = torch.load(p.model_path_prefix + "_train.pt")
    # print(train_status)
    model_filename = '%s_%02d.pt' % (p.model_path_prefix, train_status['best_epoch_so_far'])  # 取Valid最好的那一轮
    # model = torch.load(model_filename)
    dataset = Dataset(p.data_path, max_src_len=p.max_src_len, max_tgt_len=p.max_tgt_len,
                      truncate_src=p.truncate_src, truncate_tgt=p.truncate_tgt)
    v = dataset.build_vocab(p.vocab_size, embed_file=p.embed_file)
    model = Seq2Seq(v, p, use_gpu=use_gpu)
    model.load_state_dict(torch.load(model_filename))
    model.to(device)
    # 卧草TMD，原来必须要用eval()固定住结果，不然每次预测的结果都会不一样!
    # model.eval()
    model.encoder.gru.flatten_parameters()
    model.decoder.gru.flatten_parameters()

    vocab = model.vocab

    src = summary.split()
    src_len = len(src) + 1  # 最后加一个EOS
    base_oov_idx = len(vocab)
    src_tensor = torch.zeros(src_len, 1, dtype=torch.long)
    if use_gpu and torch.cuda.is_available():
        src_tensor = src_tensor.cuda()  # 迁移到cuda上去之后(这个函数不是in-place的!!!不像模型)
    oov_dict = OOVDict(base_oov_idx)
    for i, word in enumerate(src):
        idx = vocab[word]
        if idx == vocab.UNK:
            idx = oov_dict.add_word(i, word)
        src_tensor[i, 0] = idx
    src_tensor[src_len - 1, 0] = vocab.EOS

    hypotheses = model.beam_search(src_tensor, None, oov_dict.ext_vocab_size, 4, min_out_len=p.min_out_len,
                                   max_out_len=10, len_in_words=p.out_len_in_words)

    to_decode = [hypotheses[0].tokens]
    decoded_batch = decode_batch_output(to_decode, vocab, oov_dict)
    return format_tokens(decoded_batch[0])


def predict_batch(summaries, use_gpu=True):
    if use_gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device('cpu')

    p = Params()
    train_status = torch.load(p.model_path_prefix + "_train.pt")
    # print(train_status)
    model_filename = '%s_%02d.pt' % (p.model_path_prefix, train_status['best_epoch_so_far'])  # 取Valid最好的那一轮
    # model = torch.load(model_filename)
    dataset = Dataset(p.data_path, max_src_len=p.max_src_len, max_tgt_len=p.max_tgt_len,
                      truncate_src=p.truncate_src, truncate_tgt=p.truncate_tgt)
    v = dataset.build_vocab(p.vocab_size, embed_file=p.embed_file)
    model = Seq2Seq(v, p)
    model.load_state_dict(torch.load(model_filename))
    model.to(device)
    # model.eval() # 为了生成候选句，则注释掉这句话
    model.encoder.gru.flatten_parameters()
    model.decoder.gru.flatten_parameters()

    vocab = model.vocab

    res = []
    for summary in summaries:
        src = summary.split()
        src_len = len(src) + 1  # 最后加一个EOS
        base_oov_idx = len(vocab)
        src_tensor = torch.zeros(src_len, 1, dtype=torch.long)
        if use_gpu and torch.cuda.is_available():
            src_tensor = src_tensor.cuda()  # 迁移到cuda上去之后(这个函数不是in-place的!!!不像模型)
        oov_dict = OOVDict(base_oov_idx)
        for i, word in enumerate(src):
            idx = vocab[word]
            if idx == vocab.UNK:
                idx = oov_dict.add_word(i, word)
            src_tensor[i, 0] = idx
        src_tensor[src_len - 1, 0] = vocab.EOS

        hypotheses = model.beam_search(src_tensor, None, oov_dict.ext_vocab_size, 4, min_out_len=p.min_out_len,
                                       max_out_len=10, len_in_words=p.out_len_in_words)

        to_decode = [hypotheses[0].tokens]
        decoded_batch = decode_batch_output(to_decode, vocab, oov_dict)
        predict_title = format_tokens(decoded_batch[0])
        res.append(predict_title)
    return res


def del_cuplicate(title):
    # 先去除字符级重复
    words = title.split()
    st = set()
    res_words = []
    for word in words:
        if word in st:
            continue
        else:
            res_words.append(word)
            st.add(word)

    return ''.join(res_words)


if __name__ == '__main__':
    summary1 = """要 立足 防 大汛 、 抗 大旱 ， 扎实 做好 防范 应对 准备 ， 加快 补齐 水利 基础 设施 短板 ， 保障 防洪 和 供水 安全 。
 国务 委员 、 国家 森林 草原 防灭火 指挥部 总指挥 、 国家 防汛 抗旱 总指挥部 总指挥 王勇 出席 会议 并 讲话 。 """

    summary2 = """冷空气 南下 ， 华北 、 黄淮 等 地 昨日 降温 剧烈 ， 降温 幅度 超过 20℃ ， 京津冀地区 还 伴有 中到大雨 ， 体感 寒凉 。 
    预计 今天 ， 华北 大部 降雨 停歇 ， 江淮 、 江汉 等 地 的 气温 还 将 有 小幅 下降 。 昨日 ， 南方 地区 仍 以 分散性 暴雨 为主 ， 局地 
    出现 雷暴 大风 、 冰雹 等 强对流天气 ， 其中 广西 桂林 昨夜 出现 飑线 。 预计 今天 ， 江南 、 华南 等 地 仍 多发 强对流 天气 ， 浙江 、 
    福建 等地 雨势 较强 。"""

    summary3 = """昨日 ， 浙江 青田县 出现 强 雷电 、 局地 短时 强 降水 和 8-12 级 雷雨 大风 等 强 对流 天气 。 雷雨 大风 导致 东源镇 
    9 个 行政村 受灾 较为 严重 ， 辖区 部分 交通 、 供电 暂时 中断 。 房屋 屋顶 瓦片 吹落 ， 大树 被 连根 拔起 。 """

    for i in range(3):
        res = predict_batch([summary1, summary2, summary3])
        for i, title in enumerate(res):
            # print(title)
            # print('title for summary %d    ' % (i+1), title)
            print("title for summary %d    " % (i+1), del_cuplicate(title))

# 第一: 增大神经元参数
# 第二: 增加去停用词步骤
