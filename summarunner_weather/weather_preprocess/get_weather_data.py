import json

lines = []
with open('weather_511_segment.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        tmp = json.loads(line)
        content = tmp['content'].replace('  ', ' ')
        title = tmp['title'].replace('  ', ' ')
        lines.append(json.dumps({'content': content, 'title': title}, ensure_ascii=False))

with open('weather_preprocess.txt', 'w', encoding='utf-8') as f:
    f.writelines('\n'.join(lines))
