import json
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# from matplotlib.font_manager import _rebuild
#
# _rebuild() #reload一下
mpl.use('agg')

plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False # 解决负号'-'显示为方块的问题

train_final_lozz = []
val_final_lozz = []
test_lozz = []
train_lozzz = []
val_lozzz = []
with open('train_k_fold_AttnRNN_info.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        tmp = json.loads(line)
        train_lozz = tmp['train_lozz']
        train_lozzz.append(train_lozz)
        train_final_lozz.append(train_lozz[-1])
        val_lozz = tmp['val_lozz']
        val_lozzz.append(val_lozz)
        val_final_lozz.append(val_lozz[-1])
        test_loss = tmp['test_loss']
        test_lozz.append(test_loss)

df = pd.DataFrame({'train_lozz': train_final_lozz, 'val_lozz': val_final_lozz, 'test_lozz': test_lozz})
print(df)
print(df.describe())

mean_train_lozz = []
for i in range(len(train_lozzz[0])):
    num = 0
    for j in range(10):
        num += train_lozzz[j][i]
    mean_train_lozz.append(num / 10)

mean_val_lozz = []
for i in range(len(val_lozzz[0])):
    num = 0
    for j in range(10):
        num += val_lozzz[j][i]
    mean_val_lozz.append(num / 10)


plt.title(u'损失下降曲线')
# plt.plot(mean_train_lozz, color='green', label=u'训练损失')
plt.plot(mean_val_lozz, color='red', label=u'验证损失')
plt.xlabel(u'迭代轮数')
plt.ylabel(u'BCE损失')
plt.show()
plt.savefig('AttnRNN_val.png') # 保存图片
