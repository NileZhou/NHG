import json

from .clean_text import TextCleaner

"""
content词数: 100 - 800
title词数: 3 - 15
"""

cleaner = TextCleaner()
print('开始清理all_the_news')
lines = []
with open('all_the_news.txt', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f.readlines()):
        tmp = json.loads(line)
        content = tmp['content']
        title = tmp['title']
        content = cleaner.clean_text(content)
        if len(content.split()) < 100 or len(content.split()) > 800: continue
        title = cleaner.clean_text(title)
        if len(title.split()) < 3 or len(title.split())> 15: continue
        tmp = json.dumps({'content': content, 'title': title}, ensure_ascii=False)
        lines.append(tmp)
        if (i + 1) % 1000 == 0:
            print('{} lines have beed dealed'.format(i + 1))
print('现在开始清理bytecup数据')
with open('bytecup18_clean.txt', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f.readlines()):
        tmp = json.loads(line)
        content = tmp['content']
        title = tmp['title']
        content = cleaner.clean_text(content)
        if len(content.split()) < 100 or len(content.split()) > 800: continue
        title = cleaner.clean_text(title)
        if len(title.split()) < 3 or len(title.split())> 15: continue
        tmp = json.dumps({'content': content, 'title': title}, ensure_ascii=False)
        lines.append(tmp)
        if (i + 1) % 1000 == 0:
            print('{} lines have beed dealed'.format(i + 1))
print('开始写入数据')
with open('cont2tit.txt', 'w', encoding='utf-8') as f:
    f.writelines('\n'.join(lines))
print('写入{}条数据成功'.format(len(lines)))


