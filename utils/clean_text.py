import re


class TextCleaner:
    """
        处理思路:
            先全部转化为小写
            把-lrb- -rrb- 这些干了
            上正则表达式把<s>, <p>, <a>这些干了
            # 用正则表达式洗专有字符序列(邮箱, 手机号, 电话号码, url地址等) (此步骤存疑)
            同时有左右尖括号的不要, 提示可能存在html标签
            替换掉非法字符
            检查是否还存在非法字符，若还有，弃掉文章并提示有非法字符
            分词(注意和句子粘在一起的一些标点符号的处理(要么中间加空格要么直接变成空格), 同时注意\'与\"的特殊性, 同时不要空行)
            去掉多余的\n,同时也要加\n
            消除无用的引号与'``'
            消除shit...***词中*之间的空格
            清除数字逗号之间的空格
    """
    def __init__(self, replace_dict_path='replace_dict.txt'):
        self.spa_pattern = re.compile(r'</?s>|</?p>|</?a>')
        self.rb_pattern = re.compile(r'-lrb-.+?-rrb-')
        self.char_pattern = re.compile(r'[a-z|A-Z|0-9]')
        self.punct_pattern = re.compile(r'[^a-z0-9\n]')
        self.star_pattern = re.compile(r' ?\* ')
        self.number_patter = re.compile(r'\d+ [, \d+]+') # 圆括号代表分组匹配，组之间的匹配结果是或的关系，相当于 '|', 中括号是且关系
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
            print('null string')
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
        # 处理标点符号('\n'不算标点符号，等到最后再决定增删'\n')
        tokens = sent.split(' ')
        res = []
        for token in tokens:
            if token == ' ': continue
            if token == '\n':
                res.append(token)
                continue
            # 判断是否有标点符号
            puncts = self.punct_pattern.findall(token)
            if not len(puncts):
                res.append(token)
            else:
                for punct in puncts:
                    token = token.replace(punct, ' '+punct+' ')
                items = token.split()
                res += items
        sent = ' '.join(res)
        # 去掉无用的空格
        sent = self.useless_pattern.sub('', sent)
        # 处理'\n'
        sent = sent.replace('!', '!\n').replace('.', '.\n').replace(';', ';\n')
        res = []
        for sent in sent.split('\n'):
            if len(sent) > 1:
                res.append(sent)
        sent = '\n'.join(res)
        # 消除shit...***词中*之间的空格
        sent = self.star_pattern.sub('*', sent)
        # 清除数字逗号之间的空格
        matchs = self.number_patter.findall(sent)
        if len(matchs):
            # print(matchs)
            for item in matchs:
                item2 = item.replace(' , ', '')
                sent = sent.replace(item, item2)

        return sent

    def clean_text(self, text):
        # text即多个句子
        res = []
        for sent in text.split('\n'):
            if sent in ('\n', ' '): continue
            sent = self.clean_sentence(sent)
            if len(sent): res.append(sent)
        return '\n'.join(res)


if __name__ == '__main__':
    text = """-lrb- cnn -rrb- the only thing crazier than a guy in snowbound massachusetts boxing up the powdery white stuff and offering it for sale online ? people are actually buying it . for $ 89 , self-styled entrepreneur kyle waring will ship you 6 pounds of boston-area snow in an insulated styrofoam box -- enough for 10 to 15 snowballs , he says . 
    but not if you live in new england or surrounding states . `` we will not ship snow to any states in the northeast ! '' says waring 's website , shipsnowyo.com . `` we 're in the business of expunging snow ! '' his website and social media accounts claim to have filled more than 133 orders for snow -- more than 30 on tuesday alone , his busiest day yet . with more than 45 total inches , boston has set a record this winter for the snowiest month in its history . most residents see the huge piles of snow choking their yards and sidewalks as a nuisance , but waring saw an opportunity . according to boston.com , it all started a few weeks ago , when waring and his wife were shoveling deep snow from their yard in manchester-by-the-sea , a coastal suburb north of boston . he joked about shipping the stuff to friends and family in warmer states , and an idea was born . his business slogan : `` our nightmare is your dream ! '' at first , shipsnowyo sold snow packed into empty 16.9-ounce water bottles for $ 19.99 , but the snow usually melted before it reached its destination . so this week , waring began shipping larger amounts in the styrofoam cubes , which he promises will arrive anywhere in the u.s. in less than 20 hours . he also has begun selling a 10-pound box of snow for $ 119 . many of his customers appear to be companies in warm-weather states who are buying the snow as a gag , he said . whether waring can sustain his gimmicky venture into the spring remains to be seen . but he has no shortage of product . `` at this rate , it 's going to be july until the snow melts , '' he told boston.com . `` but i 've thought about taking this idea and running with it for other seasonal items . maybe i 'll ship some fall foliage . ''"""
    cleaner = TextCleaner()
    res = cleaner.clean_text(text)
    print(res)

