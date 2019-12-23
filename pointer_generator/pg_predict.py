import torch
from pointer_generator.params import Params
from pointer_generator.utils import OOVDict, format_tokens, Dataset
from pointer_generator.test import decode_batch_output
from pointer_generator.model import Seq2Seq


class PGPredictor:
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu
        if use_gpu:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device('cpu')
        p = Params()
        self.p = p
        train_status = torch.load(p.model_path_prefix + "_train.pt")
        # print(train_status)
        model_filename = '%s_%02d.pt' % (p.model_path_prefix, train_status['best_epoch_so_far'])  # 取Valid最好的那一轮
        # model = torch.load(model_filename)
        dataset = Dataset(p.data_path, max_src_len=p.max_src_len, max_tgt_len=p.max_tgt_len,
                          truncate_src=p.truncate_src, truncate_tgt=p.truncate_tgt)
        v = dataset.build_vocab(p.vocab_size, embed_file=p.embed_file)
        self.model = Seq2Seq(v, p, use_gpu=use_gpu)
        self.model.load_state_dict(torch.load(model_filename))
        self.model.to(device)
        self.model.eval()
        self.model.encoder.gru.flatten_parameters()
        self.model.decoder.gru.flatten_parameters()

        self.vocab = self.model.vocab

    def pg_predict(self, summary):
        src = summary.split()
        src_len = len(src) + 1  # 最后加一个EOS
        base_oov_idx = len(self.vocab)
        src_tensor = torch.zeros(src_len, 1, dtype=torch.long)
        if self.use_gpu and torch.cuda.is_available():
            src_tensor = src_tensor.cuda()  # 迁移到cuda上去之后(这个函数不是in-place的!!!不像模型)
        oov_dict = OOVDict(base_oov_idx)
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


def pg_predict_one(summary, use_gpu=True):
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
    model.eval()
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
    model.eval()
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


if __name__ == '__main__':

    summary1 = """the head of a hugely popular north korean girl band crossed the heavily fortified border into south 
    korea on sunday to check preparations for rare performances by an art troupe she also leads during next month'res 
    winter olympics . appearing live on south korean television, hyon song"""

    summary2 = """huawei is one of the best tech companies in the planet. it'res products are at the heart of a clash 
    between the united states and china that is straining longstanding alliances and may define the future of the 
    internet . the company started out small, selling cheap telephone switches in the 1980s in rural china . the story 
    of how it grew into a business with annual revenue of more than $100 billion is one of naked ambition, government 
    support and questionable business tactics."""

    summary3 = """the women were arrested last may and charged with crimes such as spying and undermining national 
    security. they had been campaigning for an end to the country'res male guardianship system and for the right to drive,
     before the ban was lifted last june. since then, horrific details have emerged of their alleged mistreatment at the
      hands of the Saudi authorities . on tuesday, walid al-hathloul, the brother of one of the best-known activists, 
      loujain al-hathloul, told the bbc his sister was so traumatised by what had happened to her that she wanted to 
      remain in jail, afraid of how her reputation had been unfairly smeared in her absence ."""

    english = True
    # print(predict_one(res))
    # print(predict_one(s2))
    titles = predict_batch([summary1, summary2, summary3])
    for i, title in enumerate(titles):
        print("title for summary{}: ".format(i) + title)