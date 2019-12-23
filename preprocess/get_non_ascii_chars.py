import json
from collections import Counter

cnter = Counter()

def count_txt_non_chars(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
            for c in line:
                if ord(c) > 127:
                    cnter[ord(c)] += 1
            if (i+1) % 1000 == 0: print(i+1)

count_txt_non_chars('all_the_news.txt')
count_txt_non_chars('bytecup18_clean.txt')
count_txt_non_chars('cnn_dm_txt/train.txt')
count_txt_non_chars('cnn_dm_txt/val.txt')

with open('non_ascii_chars.txt', 'w', encoding='utf-8') as f:
    f.writelines('\n'.join(sorted([str((chr(k), v)) for (k, v) in cnter.items()], reverse=True)))
