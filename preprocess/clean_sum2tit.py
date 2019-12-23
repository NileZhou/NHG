import json
import re

css_patttern = re.compile(r'\.\{.*?\}:\{.*?\} \{.*?\}')

news_prefix = ['(ap)', '(afp)', '(wjz)', '(cbs sf)', '(dpa)', '(reuters)', '(cnn)',
               '(bbc)', '(ansa)', '(spain)', '(cbs)', '(xinhua)', '(cbsla)', '(wcco)', '\n\n']

news_suffix = ['- la times', '- the new york times', '- breitbart', '- early team-news', '- food news', '- sources',
               '- report', '- live']

useless_chars = ['(photos)', '(video)', '(audio)', '(fb)', '(opinion)', '(read details)', '(md)', '\\xa0']

lines = []
with open('/home/nile/Downloads/sum2tit.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        tmp = json.loads(line)
        summary = tmp['summary']
        title = tmp['title']
        if '( )' in summary or '( )' in title: continue # 这个会极大影响句子意思
        # 替换掉无用字符
        for useless in useless_chars:
            if useless in summary:
                summary = summary.replace(useless, '')
            if useless in title:
                title = title.replace(useless, '')
        summary = css_patttern.sub('', summary) # 清理css
        # 清理报刊前缀
        summa_words = summary.split()
        summa_words_5 = summa_words[:5]
        for pref in news_prefix:
            if pref in summa_words_5:
                summa_words = summa_words[summa_words_5.index(pref)+1:]
        # 清理title后缀
        for suf in news_suffix:
            if suf in title:
                title = title.replace(suf, '')

        summary = ' '.join(summa_words)
        lines.append(json.dumps({'summary': summary, 'title': title}, ensure_ascii=False))

with open('/home/nile/Downloads/sum2tit_clean.txt', 'w', encoding='utf-8') as f:
    f.writelines('\n'.join(lines))

print('remain: ', len(lines))


