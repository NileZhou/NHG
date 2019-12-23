import os
import jieba
import json
import xml.etree.cElementTree as ET
from sklearn.model_selection import KFold


def get_data_from_xml(dir_path="/home/nile/Downloads/weather_data"):
    data = []
    file_names = os.listdir(dir_path)
    for file_name in file_names:
        tree = ET.ElementTree(file=os.path.join(dir_path, file_name))
        root = tree.getroot()
        assert root[0].tag == 'title'
        title = root[0].text
        assert root[2].tag == 'content'
        sents = [sent.text for sent in root[2].iter('sentence')]
        content = '\n'.join(sents)
        data.append(json.dumps({"content": content, "title": title}, ensure_ascii=False))

    with open('weather_origin.txt', 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(data))


def load_origin_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return lines


def preprocess(text):
    """
    目前只做了分词
    :param text:
    :return:
    """
    sents = []
    for sent in text.split('\n'):
        seg_list = jieba.cut(sent, cut_all=False)
        sents.append(' '.join(seg_list))
    return '\n'.join(sents)


def write_data(data, to_file_path='weather_511_segment.txt'):
    """
    将预处理后的数据写入硬盘
    :param to_file_path:
    :param data:
    :return:
    """
    new_data = []
    for line in data:
        tmp = json.loads(line)
        content = tmp['content']
        content = preprocess(content)
        title = tmp['title']
        title = preprocess(title)
        new_data.append(json.dumps({'content': content, 'title': title}, ensure_ascii=False))

    with open(to_file_path, 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(new_data))


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return lines


def gen_k_fold_data(data):
    X = [line['content'] for line in data]
    y = [line['title'] for line in data]
    i = 0
    for train_index, test_index in KFold(n_splits=10, random_state=0, shuffle=True).split(X, y):
        # train_X = [X[i] for i in train_index]
        # test_X = [X[i] for i in test_index]
        # train_y = [y[i] for i in train_index]
        # test_y = [y[i] for i in test_index]
        train_data = [json.dumps({'content': X[i], 'title': y[i]}, ensure_ascii=False) for i in train_index]
        test_data = [json.dumps({'content': X[i], 'title': y[i]}, ensure_ascii=False) for i in test_index]

        with open('train%d.json' % i, 'w', encoding='utf-8') as f:
            f.writelines('\n'.join(train_data))

        with open('val%d.json' % i, 'w', encoding='utf-8') as f:
            f.writelines('\n'.join(test_data))

        i += 1


def eliminate_space(data):
    data = [tmp['content'] for tmp in data]
    sents = []
    for cont in data:
        tmp_sents = cont.split('\n')
        tmp_sents = [' '.join(sent.split()) for sent in tmp_sents]
        sents += tmp_sents

    with open('weather_preprocess.txt', 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(sents))


if __name__ == '__main__':
    if not os.path.exists('weather_origin.txt'):
        get_data_from_xml()

    # data = load_origin_data('weather_origin.txt')
    data = load_data('weather_511_segment.txt')
    data = [json.loads(tmp) for tmp in data]
    # gen_k_fold_data(data)





