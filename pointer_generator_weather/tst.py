import json
import rouge


# with open('../data/chinese/sum2tit/word2id.json') as f:
#     word2id = json.load(f)
#
# evaluator = rouge.Rouge(metrics=['rouge-l'])
# with open('../data/chinese/sum2tit/sum2tit.txt', 'r', encoding='utf-8') as f:
#     for line in f.readlines():
#         tmp = json.loads(line)
#         summary = tmp['summary']
#         title = tmp['title']
#         seqs = summary.split('\n')
#         print(title)
#         for seq in seqs:
#             hyp = []
#             for word in seq.split():
#                 hyp.append(str(word2id[word]))
#             hyp = ' '.join(hyp)
#
#             ts = []
#             for word in title.split():
#                 ts.append(str(word2id[word]))
#             ref = ' '.join(ts)
#
#             scores = evaluator.get_scores(hyp, ref)
#             print(seq)
#             print(scores)
#
#         break


hyp = '0 1 2'
ref = '1 3 2'
evaluator = rouge.Rouge(metrics=['rouge-l', 'rouge-n'], max_n=2)
res = evaluator.get_scores(hyp, ref)
rouge1 = res['rouge-1']['f']
rouge2 = res['rouge-2']['f']
rougel = res['rouge-l']['f']
print(rouge1, rouge2, rougel)
