import json
import numpy as np
import math
import heapq
from summarunner_weather.utils.sim_rouge import SimRouge
from multiprocessing import Pool
from itertools import chain
# import cProfile
# import pstats


def run_proc(data, my_rouge:SimRouge, max_summa_len=4, min_sim_score=0.4):
    cont2sums = []
    sum2tits = []
    del_cnt = 0

    for line in data:
        # 有可能有些数据找不到summary(title和content一个相同词都没有)
        tmp = json.loads(line)
        content = tmp['content'].strip()
        title = tmp['title'].strip()

        seqs = [seq.strip() for seq in content.split('\n') if len(seq) > 3]  # 一句话至少得4个字符
        content = '\n'.join(seqs) # content也改变了
        if len(seqs) < max_summa_len:
            summary_list = list(range(len(seqs)))
        else:
            for i, seq in enumerate(seqs):
                words = seq.strip().split()
                seqs[i] = ' '.join(words)
            summary_list = [] # 记录summary在原文中的index
            for i, seq in enumerate(seqs):
                score = my_rouge.compute(seq, title, True) # 如果遇到词典中没有的词，将其替换为UNK_TOKEN
                if score > min_sim_score: # 最低限度达不到，说明没啥相关性
                    summary_list.append((i, score / len(seq.split()))) # 在满足总分情况下，强调"分数性价比" (过于冗长的也不要)!
            summary_list = [tup[0] for tup in heapq.nlargest(max_summa_len, summary_list, key=lambda tup: tup[1])]
            summary_list.sort()
        labels = []
        for i in range(len(seqs)):
            if i in summary_list:
                labels.append(1)
            else:
                labels.append(0)
        labels = '\n'.join([str(i) for i in labels])
        summary = '\n'.join([seqs[index] for index in summary_list])
        if not len(summary_list): # 此条数据无法抽出
            del_cnt += 1
            continue

        cont2sum = json.dumps({'content': content, 'labels': labels, 'summary': summary}, ensure_ascii=False)
        sum2tit = json.dumps({'summary': summary, 'title': title}, ensure_ascii=False)

        cont2sums.append(cont2sum)
        sum2tits.append(sum2tit)

    print('无法通过贪心算法抽出摘要的数据条数: ', del_cnt)

    return cont2sums, sum2tits


def create(origin_file_path, word2id_path, embed_path, max_summa_len=4, n_worker=1):
    """
    :param origin_file_path: 训练数据集(content->title)
    :param word2id_path: word2id文件的路径
    :param embed_path: embedding文件的路径
    :param max_summa_len: 最多抽取多少句话成为摘要
    :return:
    """

    with open(word2id_path, 'r', encoding='utf-8') as f:
        word2id = json.load(f)

    embed = np.load(embed_path)['embedding']
    my_rouge = SimRouge(word2id=word2id, embed=embed)

    with open(origin_file_path, 'r', encoding='utf-8') as f:
        data = f.readlines()

    n_chunk = math.ceil(len(data) / n_worker) # 向上取整
    p = Pool()
    chunks = [data[i: min(i+n_chunk, len(data))] for i in range(0, len(data), n_chunk)]
    res = []
    for i in range(n_worker):
        res.append(p.apply_async(run_proc, args=(chunks[i], my_rouge, max_summa_len)))

    p.close()
    p.join()
    res = [r.get() for r in res]
    cont2sumzz = [tup[0] for tup in res]
    sum2titzz = [tup[1] for tup in res]
    cont2sums = list(chain(*cont2sumzz))
    sum2tits = list(chain(*sum2titzz))

    with open('../../data/chinese/cont2sum/big/cont2sum.txt', 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(cont2sums))

    with open('../../data/chinese/cont2sum/big/sum2tit.txt', 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(sum2tits))


def main():
    # create('/home/nile/Downloads/cont2tit.txt')
    origin_file_path = '../../data/chinese/cont2sum/big/cont2tit.txt'
    word2id_path = '../../data/chinese/cont2sum/big/word2id.json'
    embed_path = '../../data/chinese/cont2sum/big/embedding.npz'

    create(origin_file_path, word2id_path, embed_path, n_worker=8)


if __name__ == "__main__":

    # cProfile.run('main()', 'restats')  # 把 cProfile 的结果输出
    # p = pstats.Stats('restats')  # pstats 读取输出的结果
    # p.sort_stats('cumulative').print_stats()  # 按照 cumtime 排序, print_stats(n) 则显示前 n 行

    main()



