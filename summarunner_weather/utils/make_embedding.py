import numpy as np
import json

# embed = np.load('data/embedding.npz')['embedding']


def from_txt_embedding_gen_npz_json(file_path='/home/nile/Downloads/ft (1)'):
    words = ['PAD_TOKEN', 'UNK_TOKEN']
    vecs = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
            if i == 0:
                print(line)
                continue
            word = line.split('\t')[0]
            words.append(word)
            nums = line.strip().split('\t')[1]
            vec = [float(num) for num in nums.split()]
            vecs.append(vec)
    zeros = np.zeros(shape=(2, 100), dtype=np.float32) # for pad and unk
    vecs = np.array(vecs, dtype=np.float32)
    vecs = np.vstack((zeros, vecs))
    np.savez('/media/nile/study/repositorys/autosumma/data/chinese/cont2sum/big/embedding.npz', embedding=vecs)
    with open("/media/nile/study/repositorys/autosumma/data/chinese/cont2sum/big/word2id.json", 'w') as f:
        json.dump(dict(zip(words, list(range(len(words))))), f, ensure_ascii=False)


from_txt_embedding_gen_npz_json()
#
# embed = np.load('embedding.npz')['embedding']
# print(embed.shape)
# print(embed[0])
# print(embed[2])
#
# with open('word2id.json') as f:
#     word2id = json.load(f)
#
# print(word2id['UNK_TOKEN'])
# print(word2id['中国'])

