import json


def clean(file_path, *col_names, max_word_len=20):
    lines = []
    del_cnt = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            tmp = json.loads(line)
            col1, col2 = col_names[0], col_names[1]
            text = tmp[col1]
            words = text.split()
            m_word_len = max([len(word) for word in words])
            text = tmp[col2]
            words = text.split()
            s_word_len = max([len(word) for word in words])
            max_w_len = max(m_word_len, s_word_len)
            if max_w_len > max_word_len:
                del_cnt += 1
                continue
            lines.append(line)

    with open(file_path[:-4] + "_clean.txt", 'w', encoding='utf-8') as f:
        f.writelines(''.join(lines))

    print('clean ' + file_path + ' success')
    print('delete count: ', del_cnt)
    print('reamin count: ', len(lines))


clean('cont2sum.txt', 'content', 'summary')
clean('cont2tit.txt', 'content', 'title')
clean('sum2tit.txt', 'summary', 'title')
