import rouge


metrics = ['rouge-n', 'rouge-l', 'rouge-w']

evaluator = rouge.Rouge(metrics=metrics,
                        max_n=3, # 如果有metrics里有rouge-n这一项, 这就是最大的n. 比如max_n = 2, 则有rouge-1, rouge-2
                        alpha=0.5, # default F1 score
                        weight_factor=1.2, # 计算rouge-l与rouge-w时， precision = precision ** (1.0 / weight_factor)
                        stemming=True)

hypothesis = "the cat was found under the bed" # 自动生成的摘要
reference = "the cat was under the bed" # 人工生成的参考摘要

# rouge关注的是召回率(recall)
# rouge-1: 6/6
# rouge-2: (the cat) (cat was) (was under) (under the) (the bed) 这5个在hypothesis中出现了4个,所以为4/5
# rouge-3: (the cat was) (cat was under) (was under the) (under the bed) 2/4
# rouge-l: LCS(longest common subsequence)(最长公共子序列而不是最长公共子串) R = \frac{LCS(X, Y)}{model}, P = \frac{LCS(X, Y)}{n}
# 其中m为reference(Y)的长度, n为hypothesis(X)的长度(所含词的个数) 3 / 6, 不过要注意weight_factor的存在(加权最长公共子序列)


# 输出的'params'代表precision, 'r'代表recall(即我们关心的), 'f'代表f1-score

scores = evaluator.get_scores(hypothesis=hypothesis.lower(), references=reference.lower())
for k, v in scores.items():
    print(k, v)

# print(scores['rouge-1']['r'])
# print(scores['rouge-2']['r'])
# print(scores['rouge-3']['r'])

