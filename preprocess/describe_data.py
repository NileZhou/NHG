import json
import pandas as pd


def describe(file_path, *cols):
    col1, col2 = cols[0], cols[1]
    text1_word_num = []
    text2_word_num = []
    text1_char_num = []
    text2_char_num = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            tmp = json.loads(line)
            text1 = tmp[col1]
            text2 = tmp[col2]
            text1_words = text1.split()
            text1_word_num.append(len(text1_words))
            text1_char_num.append(len(text1))
            text2_words = text2.split()
            text2_word_num.append(len(text2_words))
            text2_char_num.append(len(text2))

    df = pd.DataFrame({col1+" word num" : text1_word_num, col1+ ' char num': text1_char_num, col2+' word num':
        text2_word_num, col2+' char num': text2_char_num})
    print(file_path, ' describe: ')
    print(df.describe())


if __name__ == "__main__":
    # describe('cont2sum.txt', 'content', 'summary')
    # describe('cont2tit.txt', 'content', 'title')
    describe('/home/nile/Downloads/sum2tit.txt', 'summary', 'title')
