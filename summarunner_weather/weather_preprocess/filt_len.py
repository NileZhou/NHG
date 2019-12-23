import json
import pandas as pd


con_len = []
tit_len = []
lines = []
del_cnt = 0
with open('weather_preprocess.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        tmp = json.loads(line)
        content = tmp['content']
        title = tmp['title']
        if len(content.split()) < 100 or len(content.split()) > 800:
            del_cnt += 1
            continue
        if len(title.split()) > 15:
            del_cnt += 1
            continue
        lines.append(json.dumps({'content': content, 'title': title}, ensure_ascii=False))
        con_len.append(len(content.split()))
        tit_len.append(len(title.split()))

with open('filted_weather.txt', 'w', encoding='utf-8') as f:
    f.writelines('\n'.join(lines))
df = pd.DataFrame({'con_len': con_len, 'tit_len': tit_len})
print(df.describe())
print("delete cnt: ", del_cnt)