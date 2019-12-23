res = """Building prefix dict from the default dictionary ...
Loading model from cache /tmp/jieba.cache
Loading model cost 0.605 seconds.
Prefix dict has been built succesfully.
预测中文标题
Reading dataset ../data/chinese/sum2tit/train.txt... 2351 pairs.
../data/chinese/sum2tit/train_10000_vocab  already exist.
/home/nile/anaconda3/lib/python3.6/site-packages/torch/nn/modules/rnn.py:46: UserWarning: dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1, but got dropout=0.5 and num_layers=1
  "num_layers={}".format(dropout, num_layers))
原始气象词表(去除各种英文数字符号后的词表)大小:  7694
num of 1-gram:  7493
num of 2-gram:  3369957
num of 3-gram:  15036464
5 月 6 日 ， 广州 超大 城市 综合 气象观测 试验 2019 年 增强 观测 期 启动 会 召开 。
按照 中国气象局 部署 ， 广东省 气象局 将 开展 为期 三年 的 超大 城市 综合 气象观测 试验 ， 致力于 解决 大城市 临近 预报 和 环境 气象 服务 中 关键性 核心技术 问题 。
2019 年 广州 超大 城市 综合 气象观测 试验 ， 将 开展 大气 综合 廓线 站 观测网 建设 和 进行 增强 期 观测 试验 。
==============================
广东省综合气象观测城市期"""

summary = res[res.index('15036464') + 9: res.index('==============================')]
print(summary)
title = res[res.index('==============================') + 30:]
print(title)