import os
import re
import jieba
import argparse
from preprocess.clean_text import TextCleaner
from nltk import sent_tokenize
from summarunner.summarunner_predict import summa_predict
from summarunner_weather.predict import summa_weather_predict
from pointer_generator.pg_predict import PGPredictor
from pointer_generator_weather.pgw_predict import PGWPredictor
from pointer_generator_weather.pgw_predict import del_cuplicate

non_chinese_pat = re.compile(r'[a-zA-Z0-9０１２３４５６７８９，、。？“”！：；‘’《》%…·]+')


class TextLengthError(Exception):
    def __init__(self, error_info):
        super().__init__(self)  # 先初始化父类
        self.error_info = error_info

    def __str__(self):
        return self.error_info


class ModelNameNotFoundError(Exception):
    def __init__(self, error_info):
        super().__init__(self)
        self.error_info = error_info

    def __str__(self):
        return self.error_info


def chinese_sent_tokenize(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")


def clean_string(origin_s, is_english):
    if is_english:
        words = origin_s.split()
        if len(words) < 100:
            raise TextLengthError('the length of content is smaller than 100, it\' illegal.')
        elif len(words) > 800:
            raise TextLengthError('the length of content is larger than 800, it\'res illegal.')
        else:
            cleaner = TextCleaner(replace_dict_path=os.path.join('preprocess', 'replace_dict.txt'))
            return '\n'.join(sent_tokenize(cleaner.clean_text(origin_s), language='english'))
    else:
        clean_sents = chinese_sent_tokenize(origin_s.lower())  # 中文也要转换为小写
        sents = []
        for sent in clean_sents:
            seg_list = jieba.cut(sent, cut_all=False)
            sents.append(' '.join(seg_list))
        s = '\n'.join([sent.strip() for sent in sents])
        s = re.sub('\n+', '\n', s)
        words = s.split()
        if len(words) < 100:
            raise TextLengthError('文本长度小于100个词，太少了!')
        elif len(words) > 800:
            raise TextLengthError('文本长度大于800个词，太多了!')
        else:
            return s


def get_summary(content, is_english, model_prefix, use_gpu=True):
    assert model_prefix in ('AttnRNN', 'CNN_RNN', 'RNN_RNN')
    if is_english:
        load_path = os.path.join('summarunner', 'checkpoints', model_prefix)
        embedding_path = os.path.join('data', 'english', 'cont2sum', 'embedding.npz')
        word2id_path = os.path.join('data', 'english', 'cont2sum', 'word2id.json')
        summary = summa_predict(content, load_path, embedding_path, word2id_path, 800, 60, 3, use_gpu=use_gpu)
        return summary
    else:
        load_path = os.path.join('summarunner_weather', 'little', 'checkpoints', model_prefix)
        embedding_path = os.path.join('data', 'chinese', 'cont2sum', 'little', 'embedding.npz')
        word2id_path = os.path.join('data', 'chinese', 'cont2sum', 'little', 'word2id.json')
        summary = summa_weather_predict(content, load_path, embedding_path, word2id_path, 800, 70, 3, use_gpu=use_gpu)
        return summary


def get_title(summary, is_english, use_gpu=False):
    if is_english:
        os.chdir('pointer_generator')
        pg_predictor = PGPredictor(use_gpu=use_gpu)
        ititle = pg_predictor.pg_predict(summary)
        print(ititle)
    else:
        os.chdir('pointer_generator_weather')
        print('预测中文标题')
        pgw_predictor = PGWPredictor(use_gpu=use_gpu)
        # language_model = WeatherTriGram()
        imax = -2**31
        ititle = ''
        for i in range(10):
            title = pgw_predictor.pgw_predict(summary)
            title = del_cuplicate(title)
            title = re.sub(non_chinese_pat, '', title)
            # score = language_model.compute_sentence_prob(' '.join(jieba.cut(title)))
            # if not score:
            #     # print('Warning! score is NoneType')
            #     ititle = ''
            # if not score: ititle = ''
            # elif score > imax:
            #     imax = score
            #     ititle = title
            ititle = title
            # print('socre: ', score, ' \ntitle: ', title)
    os.chdir('../')

    # 这里要改变为选取候选句最好的title
    return ititle


def pipeline(article, model_prefix, use_gpu=True):
    # 先判断是英文还是中文
    ascii_cnt = 0
    unascii_cnt = 0
    for c in article:
        if ord(c) < 128: ascii_cnt += 1
        else: unascii_cnt += 1
    is_english = ascii_cnt > unascii_cnt
    if is_english:
        summary = get_summary(content=clean_string(article, True), is_english=True, model_prefix=model_prefix,
                              use_gpu=use_gpu)
        title = get_title(summary, True, use_gpu=use_gpu)
    else:
        summary = get_summary(clean_string(article, False), False, model_prefix, use_gpu=use_gpu)
        # 中文要多一步: 把摘要切分。
        res = []
        sents = summary.split('\n')
        for sent in sents:
            res.append(' '.join(jieba.cut(sent, cut_all=False)))
        summary = '\n'.join(res)
        title = get_title(summary, False, use_gpu=use_gpu)
    print('==============================')
    print(summary)
    print('==============================')
    print(title)

    return summary, title


if __name__ == '__main__':

    # 创建解析步骤
    parser = argparse.ArgumentParser(description='generate summary and headline')

    parser.add_argument('-article_path', type=str, help='the article needed to be generate')
    parser.add_argument('-model_prefix', type=str, help='the model prefix of extractive models')

    args = parser.parse_args()
    # pipeline(args.article, args.model_prefix)
    with open(args.article_path, 'r', encoding='utf-8') as f:
        article = '\n'.join(f.readlines())
    pipeline(article, args.model_prefix)


    # 单例
    # use_gpu = True
    # model_prefix = 'CNN_RNN'
    # is_english = False
    #
    # content = """WhatsApp has revealed a vulnerability in its system that could have allowed hackers access to its users' phones, with a London-based human rights lawyer possibly among the targets.
    # The encrypted messaging service, owned by Facebook (FB), said Monday that it had discovered and fixed the vulnerability the attackers had sought to exploit. The hackers could implant malicious code on a victim'res phone by placing a voice call to the victim on WhatsApp.
    # "The attack has all the hallmarks of a private company reportedly that works with governments to deliver spyware that takes over the functions of mobile phone operating systems," a WhatsApp spokesperson said in a statement.
    # While WhatsApp did not name the private company, a source familiar with the investigation into the attack said that company is NSO Group, an Israeli cyber company that has developed a powerful piece of malware designed to spy on its victims.
    # In a statement provided to CNN on Monday, NSO said, "Under no circumstances would NSO be involved in the operating or identifying of targets of its technology, which is solely operated by intelligence and law enforcement agencies."
    # NSO said its technology was licensed to government agencies "for the sole purpose of fighting crime and terror," adding that those agencies determine how the technology is used without any involvement from the company.
    # The Financial Times first reported details of the vulnerability."""
    #
    # content_chinese = """5月6日，广州超大城市综合气象观测试验2019年增强观测期启动会召开。来自中国气象局综合观测司、中国气象局气象探测中心、广东省气象局、上海市气象局、广州市科技局、广州市气象局等单位领导和专家共100余人参加启动会。
    # 　　按照中国气象局部署，广东省气象局将开展为期三年的超大城市综合气象观测试验，致力于解决大城市临近预报和环境气象服务中关键性核心技术问题。试验主要内容包括开展城市冠层观测试验；开展雷雨大风高时空分辨率、高覆盖天气雷达观测试验，构建高分辨率、高（全）覆盖天气雷达探测格点数据库；开展综合观测资料的融合性分析与应用和资料同化分析试验，构建覆盖城市的高分辨率实时三维实况场；开展强降水观测试验，提升数值模式对台风强风强降水以及沿海地区强降水的预报能力；开展灰霾观测试验，提高珠三角灰霾与空气质量预报水平。
    # 　　2019年广州超大城市综合气象观测试验，将开展大气综合廓线站观测网建设和进行增强期观测试验。目前已做好了各项准备工作，已选取了广州市局和黄埔区局2个站点开展试验，龙门站作为对比站，已完成了5条垂直廓线观测设备的布设。下来将结合典型天气过程获取温湿、风、水凝物、气溶胶5条廓线数据，通过数据分析将揭示广州超大城市及城市群对气象环境，尤其是对大气边界层的影响；并通过建立观测预报的良性互动机制试验，探索未来观测与预报一体化的业务和业务流程。"""
    #
    # # 单一预测
    # if is_english:
    #     summary = get_summary(content=clean_string(content, True), is_english=True, model_prefix=model_prefix, use_gpu=use_gpu)
    #     print('summary: ')
    #     print(summary)
    #     title = get_title(summary, True, use_gpu=use_gpu)
    # else:
    #     summary = get_summary(clean_string(content_chinese, False), False, model_prefix, use_gpu=use_gpu)
    #     print('summary: ')
    #     print(summary)
    #
    #     # 中文要多一步: 把摘要切分。
    #     res = []
    #     sents = summary.split('\n')
    #     for sent in sents:
    #         res.append(' '.join(jieba.cut(sent, cut_all=False)))
    #     summary = '\n'.join(res)
    #     print('title: ')
    #     title = get_title(summary, False, use_gpu=use_gpu)
    #     print(title)


    # from rouge import Rouge
    #
    # evaluator = Rouge(metrics=['rouge-n', 'rouge-l'], max_n=2)
    # rouge1s = []
    # rouge2s = []
    # rougels = []
    #
    # # 测试
    # if is_english:
    #     os.chdir('pointer_generator')
    #     pg_predictor = PGPredictor(use_gpu=use_gpu)
    #     with open('../data/english/sum2tit/test.txt', 'r', encoding='utf-8') as f:
    #         for i, line in enumerate(f.readlines()):
    #             if i > 0 and i % 100 == 0: print(i)
    #             if i >= 1000: break
    #             tmp = json.loads(line)
    #             summary = tmp['summary']
    #             ref_title = tmp['title']
    #             title = pg_predictor.pg_predict(summary)
    #             res = evaluator.get_scores(title, ref_title)
    #             rouge1 = res['rouge-1']['f']
    #             rouge2 = res['rouge-2']['f']
    #             rougel = res['rouge-l']['f']
    #             rouge1s.append(rouge1)
    #             rouge2s.append(rouge2)
    #             rougels.append(rougel)
    #
    # else:
    #     os.chdir('pointer_generator_weather')
    #     pgw_predictor = PGWPredictor(use_gpu=use_gpu)
    #     vocab = pgw_predictor.vocab
    #     language_model = WeatherTriGram()
    #     with open('../data/chinese/sum2tit/test.txt', 'r', encoding='utf-8') as f:
    #         for i, line in enumerate(f.readlines()):
    #             if i > 0 and i % 10 == 0: print(i)
    #             if i >= 300: break
    #             tmp = json.loads(line)
    #             summary = tmp['summary']
    #             ref_title = tmp['title']
    #             title = pgw_predictor.pgw_predict(summary)
    #
    #             # imax = -2 ** 31
    #             # ititle = ''
    #             # for i in range(10):
    #             #     title = pgw_predictor.pgw_predict(summary)
    #             #     title = del_cuplicate(title)
    #             #     title = re.sub(non_chinese_pat, '', title)
    #             #     score = language_model.compute_sentence_prob(' '.join(jieba.cut(title)))
    #             #     if not score:
    #             #         print('Warning! score is NoneType')
    #             #         ititle = ''
    #             #         continue
    #             #     if score > imax:
    #             #         imax = score
    #             #         ititle = title
    #             ititle = title
    #
    #             ref_indexies = ' '.join([str(vocab.w2i(word)) for word in ref_title.split()])
    #             title_indexies = ' '.join([str(vocab.w2i(word)) for word in ititle.split()])
    #             res = evaluator.get_scores(title_indexies, ref_indexies)
    #             rouge1 = res['rouge-1']['f']
    #             rouge2 = res['rouge-2']['f']
    #             rougel = res['rouge-l']['f']
    #             rouge1s.append(rouge1)
    #             rouge2s.append(rouge2)
    #             rougels.append(rougel)
    # df = pd.DataFrame({'rouge-1': rougels, 'rouge-2': rouge2s, 'rouge-l': rougels})
    # print(df.describe())
    # os.chdir('../')
