import json
import gc

# 先去重
ud = dict() # hash of a string -> pos
pozz = [] # store the positions
dup_cnt = 0 # record the duplicate
with open("train_100w.txt", "r", encoding='utf-8') as f:
    for i, line in enumerate(f.readlines()):
        tmp = json.loads(line)['content'] + json.loads(line)['title']
        h = hash(tmp)
        if h not in ud:
            ud[h] = i
            pozz.append(i)
        else:
            dup_cnt += 1
del ud
gc.collect()
p = 0 # record the next index of num which should be placed into new_lines
new_lines = []
# 再去掉含非法字符的文章(同时替换掉一些字符)
illegal_cnt = 0 # store the number of essay which contains illegal character
rep_d = dict() # ord(char) -> ord(char)
with open("replace_char.txt", "r", encoding='utf-8') as f:
    for line in  f.readlines():
        rep_d[ord(line[0])] = ord(line[2])
with open("train_100w.txt", "r", encoding='utf-8') as f:
    for i, line in enumerate(f.readlines()):
        if pozz[p] != i: continue
        p += 1
        # first, replace some char
        new_line = []
        for c in line:
            if ord(c) in rep_d:
                new_line.append(chr(rep_d[ord(c)]))
            else:
                new_line.append(c)
        line = ''.join(new_line)
        content = json.loads(line)['content']
        title = json.loads(line)['title']
        tmp = content + title
        rights = set([32, 33, 34, 39, 40, 41, 44, 45, 46, 61, 63, 64, 92, 95, 96, 124, 126] + list(range(48, 60)) + list(range(65, 91)) + list(range(97, 123)))
        judge = True
        for c in tmp: # check every char
            if ord(c) not in rights:
                judge = False
                break
        if judge:
            new_lines.append(line.lower()) # meantime, invoke the lower() fun
        else:
            illegal_cnt += 1

with open("train_clean.txt", 'w', encoding='utf-8') as f:
    f.writelines(''.join(new_lines))

print('dup_cnt: ', dup_cnt, "illegal_cnt: ", illegal_cnt, "remain cnt: ", len(new_lines))

# test the file
with open('train_clean.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        tmp = json.loads(line)
print('test success')
