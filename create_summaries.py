import json
from utils.my_rouge import Myrouge


def create(origin_file_path='train_3k.txt'):
    my_rouge = Myrouge()

    data = [] # (json{"summary":xxx, "title":title})

    with open(origin_file_path, 'r', encoding='utf-8') as f:
        for _, line in enumerate(f.readlines()): # 对于每一条训练数据
            # 有可能有些数据找不到summary(title和content一个相同词都没有)
            tmp = json.loads(line)
            content = tmp['content'].strip()
            title = tmp['title'].strip()
            seqs = [seq for seq in content.split('\n') if len(seq) > 3] # 至少一句话得有3个字母吧
            summary_list = []
            index_set = set()
            pre_score = 0
            # 初始化第一句
            for i, seq in enumerate(seqs):
                cur = my_rouge.compute(seq, title)
                if cur > pre_score:
                    pre_score = cur
                    if not len(summary_list):
                        summary_list.append(i)
                    else:
                        summary_list[0] = i
            if not len(summary_list):
                continue # 一个相同词都没有
            index_set.add(summary_list[0])
            while len(summary_list) <= len(seqs):
                pre_summary = '\n'.join([seqs[i] for i in summary_list])
                tmp_i = -1
                for i, seq in enumerate(seqs):
                    if i not in index_set:
                        cur = my_rouge.compute(pre_summary + '\n' + seq, title)
                        if cur > pre_score:
                            pre_score = cur
                            tmp_i = i
                if tmp_i == -1: break
                index_set.add(tmp_i)
                summary_list.append(tmp_i)
                summary_list.sort()

            summary = '\n'.join([seqs[i] for i in summary_list])
            res = json.dumps({'summary': summary, 'title': title}, ensure_ascii=False)
            data.append(res)

    data_len = len(data)

    with open('summ_t_{}.txt'.format(data_len), 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(data))


if __name__ == "__main__":
    # create('/media/nile/study/softwares/baidu_netdisk/download/bytecup2018/train_clean.txt')
    # test
    with open("summ_t_83566.txt", 'r', encoding='utf-8') as f:
        for line in f.readlines():
            tmp = json.loads(line)
            summary = tmp['summary']
            title = tmp['summary']
    print('test success')



