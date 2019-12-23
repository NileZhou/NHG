import json


"""
because we want to train model in a end-to-end way, and we want to generate the coherence sentence.
so we don't do the Lemmatization(词形还原) and Stemming(词干提取)
just filter the link and ugly-character(non-ascii character and http tags)
"""


def get_remove_duplicate_lines(origin_data, *column_names):
    """
    remove the duplicate line of data set.

    origin_data: List[json str], 原训练集/验证集数据
    column_names: str1, str2, ..., json字符串的key名字

    return: List[str]
    """
    str2pos = dict()
    pozz = []
    dup_num = 0
    total_num = 0
    for i, line in enumerate(origin_data):
        tmp = json.loads(line)
        s = ''
        for col in column_names:
            s += tmp[col].strip()
        h = hash(s)
        if h not in str2pos:
            str2pos[h] = i
            pozz.append(i)
        else:
            dup_num += 1
        total_num += 1
    # start delete
    lines = []
    p = 0
    for i, line in enumerate(origin_data):
        if p == i:
            lines.append(line)
            p += 1
    print('total number of {} : {}, duplicate number: {}, the number of remain text: {}'.format(origin_file_path,
                                                                                                total_num, dup_num,
                                                                                                len(lines)))
    return lines





