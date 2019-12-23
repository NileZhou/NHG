import json

# title rubbish pattern:
# title中'v'单独构成单词的不要
# xxx xxx fast facts
# for the record
# 'what we\'re reading'

lines = []
with open('/home/nile/Downloads/sum2tit.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        tmp = json.loads(line)
        title = tmp['title']
        if 'fast facts' in title: continue
        if 'for the record' in title: continue
        if 'what we\'re reading' in title: continue
        if 'v' in title.split(): continue
        lines.append(line)

with open('/home/nile/Downloads/sum2tit_clean.txt', 'w', encoding='utf-8') as f:
    f.writelines(''.join(lines))

print('remain: ', len(lines))