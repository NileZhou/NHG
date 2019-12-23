import sqlite3
import json


"""
I found the datasete on https://www.kaggle.com/snapcrack/all-the-news
the origin data comes from kaggle dataset(https://components.one/datasets/all-the-news-articles-dataset/)
it'res a .db file which contains 204,135 articles

now we want fetch the content info and title info into a txt file
"""
def get_data(db_file_path='all-the-news.db', to_file_path='all_the_news.txt', min_title_len=10, min_content_len=100):
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()
    try:
        data = []
        cursor.execute('select * from longform')
        cnt = 0
        for line in cursor.fetchall():
            title = line[1]
            content = line[4]
            if not line or not content:
                continue
            if len(title) < min_title_len or len(content) < min_content_len: continue
            data.append(json.dumps({'content': content.strip(), 'title': title.strip()}, ensure_ascii=False))
            cnt += 1
            if cnt % 1000 == 0: print('{} has be added into data'.format(cnt))
        with open(to_file_path, 'w', encoding='utf-8') as f:
            f.writelines('\n'.join(data))
        print('total {} lines were added into {}'.format(cnt, to_file_path))
    except Exception as ex:
        print(ex)
    finally:
        cursor.close()
        conn.close()


def get_remove_duplicate_line(origin_file_path, *column_names):
    """
    origin_file_path: str, 训练集/验证集路径
    column_names: str1, str2, ..., json字符串的key名字

    return: List[str]
    """
    str2pos = dict()
    pozz = []
    dup_num = 0
    total_num = 0
    with open(origin_file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
            tmp = json.loads(line)
            s = ''
            for col in column_names:
                s += tmp[col].strip()
            h = hash(s)
            if h not in str2pos:
                str2pos[h] = i
                pozz.append(i)
            else:
                dup_num += 1
            total_num += 1
    # start delete
    lines = []
    p = 0
    with open(origin_file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
            if p == i:
                lines.append(line)
                p += 1
    print('total number of {} : {}, duplicate number: {}, the number of remain text: {}'.format(origin_file_path,
                                                                                                total_num, dup_num,
                                                                                                len(lines)))
    return lines


if __name__ == "__main__":
    get_data()
    lines = get_remove_duplicate_line('all_the_news.txt', 'content', 'title')
    with open('all_the_news.txt', 'w', encoding='utf-8') as f:
        f.writelines(''.join(lines))
