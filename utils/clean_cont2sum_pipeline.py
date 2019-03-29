import re
import json
import re


class TextCleaner:
    """
        处理思路:
            先全部转化为小写
            把-lrb- -rrb- 这些干了
            上正则表达式把<s>, <p>, <a>这些干了
            同时有左右尖括号的不要, 提示可能存在html标签
            替换掉非法字符
            检查是否还存在非法字符，若还有，弃掉文章并提示有非法字符
            消除无用的引号与'``'
            #不让某些标点符号与单词粘在一起
            消除shit...***词中*之间的空格
            #清除数字逗号之间的空格
            去除多余的'\n'
    """

    def __init__(self, replace_dict_path='replace_dict.txt'):
        self.spa_pattern = re.compile(r'</?s>|</?p>|</?a>')
        self.rb_pattern = re.compile(r'-lrb-.+?-rrb-')
        self.char_pattern = re.compile(r'[a-z|A-Z|0-9]')
        # self.punct_pattern = re.compile(r'[^a-z0-9\n]')
        self.star_pattern = re.compile(r' ?\* ')
        self.number_patter = re.compile(r'\d+ [, \d+]+')  # 圆括号代表分组匹配，组之间的匹配结果是或的关系，相当于 '|', 中括号是且关系
        self.useless_pattern = re.compile(r'(\' *\')|(\" *\")|(` *`)')
        # url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        # email_pattern = re.compile(r'[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+')
        # phone_pattern = re.compile(r'(1[3|4|5|8][0-9]d{4,8})|(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})')
        # date_pattern = re.compile(r'\d{1,4}(-|.|\||,)\d{1,4}(-|.|\||,)\d{1,4}')
        self.replace_dict = dict()  # Dict[int, str] ord(char) -> replace chars
        with open(replace_dict_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                L = line.split()
                c, chars = L[0], L[1]
                self.replace_dict[ord(c)] = chars

    def clean_sentence(self, sent, output=False):
        if not sent or len(sent) < 1:
            if output: print('the string which we want to clean is a null string')
            return ''
        sent = sent.lower()
        sent = self.spa_pattern.sub('', sent)
        sent = self.rb_pattern.sub('', sent)
        # TODO 目前暂不清洗专有字符如邮箱手机等
        if '<' in sent and '>' in sent:
            if output:
                print('found "<" and ">" exist in the sent, might exist some illegal tag')
            return ''
        # 替换某些特定非ascii字符
        res = []
        for c in sent:
            if ord(c) in self.replace_dict:
                res.append(self.replace_dict[ord(c)])
            else:
                res.append(c)
        sent = ''.join(res)
        # 清除含non-ascii的文本
        for c in sent:
            if ord(c) > 127:
                if output:
                    print('exist non-ascii char in the sent, it\'s illegal.')
                return ''
        # 去掉无用的引号
        sent = self.useless_pattern.sub('', sent)
        # 消除星号间的空格
        for item in self.star_pattern.findall(sent):
            sent = sent.replace(item, '*')
        # 去除多余的'\n'
        res = []
        for sent in sent.split('\n'):
            if len(sent) > 1:
                res.append(sent.strip())
        sent = '\n'.join(res)

        return sent

    def clean_text(self, text, output=False):
        """
        限制词数: 文本中每句话:
        :param text:
        :return:
        """
        # text即多个句子
        res = []
        for sent in text.split('\n'):
            if sent in ('\n', ' '): continue
            sent = self.clean_sentence(sent, output)
            if len(sent): res.append(sent)
        return '\n'.join(res)


if __name__ == '__main__':
    cleaner = TextCleaner()
    print('开始清理cnn_dm_txt中的train.txt')
    lines = []
    with open('cnn_dm_txt/train.txt', 'r', encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
            tmp = json.loads(line)
            content = tmp['content']
            summary = tmp['summary']
            content = cleaner.clean_text(content)
            if not len(content): continue
            summary = cleaner.clean_text(summary)
            if not len(summary): continue
            tmp = json.dumps({'content': content, 'summary': summary})
            lines.append(tmp)
            if (i + 1) % 1000 == 0:
                print('{} lines have beed dealed'.format(i + 1))
    print('cnn_dm_txt/train.txt共余{}行'.format(len(lines)))
    print('开始清理cnn_dm_txt/val.txt中的val.txt')
    with open('cnn_dm_txt/val.txt', 'r', encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
            tmp = json.loads(line)
            content = tmp['content']
            summary = tmp['summary']
            content = cleaner.clean_text(content)
            if not len(content): continue
            summary = cleaner.clean_text(summary)
            if not len(summary): continue
            tmp = json.dumps({'content': content, 'summary': summary})
            lines.append(tmp)
            if (i + 1) % 1000 == 0:
                print('{} lines have beed dealed'.format(i + 1))
    print('现在一共有{}行待写入'.format(len(lines)))
    with open('cnn_dm_txt/content2summary.txt', 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(lines))
    print('写入完毕')
    print('开始写入规范长度数据(content 词数: 100 - 800, summary 词数: 5 - 60)')
    with open('cnn_dm_txt/cont2sum.txt', 'w', encoding='utf-8') as f:
        f.writelines('\n'.join([line for line in lines if 100 <= len(json.loads(line)['content'].split()) <= 800 and
                                5 <= len(json.loads(line)['summary'].split()) <= 60]))
    print('写入成功')


