import json
from collections import Counter


def get_grams(text, n: int):
    assert len(text) > n
    words = text.split()
    assert len(words) >= n
    res = []
    for i in range(0, len(words)-n+1):
        res.append(' '.join(words[i: i+n]))

    return res


def get_prefix_suffix(file_path, num_prefix=5, num_suffix=5, suffix_right=True, *cols):
    """
    截取json字符串第一个串的前 num_prefix 个单词与第二个串的前/后 num_suffix 个单词(取决于suffix_right是不是True)
    :param file_path:
    :param num_prefix: 要截取的text1前面的词数
    :param num_suffix: 要截取的text2后面的词数
    :param suffix_right: 如果为True,截取第二个词右边的num_suffix个词
    :param cols: 认为输入的json子级名
    :return:
    """
    pres, sufs = [], []
    col1, col2 = cols[0], cols[1]
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            tmp = json.loads(line)
            text1 = tmp[col1]
            pres.append(' '.join(text1.split()[: num_prefix]))
            text2 = tmp[col2]
            if len(text2.split()) < num_suffix:
                sufs.append(text2)
            else:
                if suffix_right:
                    sufs.append(' '.join(text2.split()[-num_suffix:]))
                else:
                    sufs.append(' '.join(text2.split()[: num_suffix]))

    return pres, sufs


def find_duplicate_n_gram(phases, max_n=5):
    """

    :param grams: 传递过来的词组
    :param max_n: 最大重复n_gram的n
    :return:
    """
    n_ptr = 2
    while n_ptr <= max_n:
        print("len phases: ", len(phases))
        cnter = Counter()
        pozz = [] # 存储存在重复gram的phase index
        for i, phase in enumerate(phases):
            if len(phase.split()) < n_ptr: continue
            grams = get_grams(phase, n_ptr)
            for gram in grams:
                cnter[gram] += 1
                if cnter[gram] > 1:
                    pozz.append(i)
        keys = list(cnter.keys())
        for k in keys:
            if cnter[k] == 1:
                del cnter[k]

        res = sorted([tup for tup in cnter.items()], key=lambda tup: tup[1], reverse=True)
        with open('duplicate_{}_grams.txt'.format(n_ptr), 'w', encoding='utf-8') as f:
            f.writelines('\n'.join([tmp[0] for tmp in res]))

        if len(pozz):
            phases = [phases[i] for i in pozz]
            n_ptr += 1
        else:
            break


# 通过n-gram与人工筛选，最终得到这些杂志、媒体、报纸名

news_prefix = ['(ap)', '(afp)', '(wjz)', '(cbs sf)', '(dpa)', '(reuters)', '(cnn)',
            '(bbc)', '(ansa)', '(spain)', '(cbs)', '(xinhua)', '(cbsla)', '(wcco)']

news_suffix = ['- la times', '- the new york times', '- breitbart', '- early team-news', '- food news', '- sources',
               '- report', '- live']

useless_chars = ['(photos)', '(video)', '(audio)', '(fb)', '(opinion)', '(read details)', '(md)']

if __name__ == '__main__':
    pass
    # pres, sufs = get_prefix_suffix('/home/nile/Downloads/sum2tit.txt', 5, 5, True, 'summary', 'title')
    # find_duplicate_n_gram(pres)
    # find_duplicate_n_gram(sufs)


