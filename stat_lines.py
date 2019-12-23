import os
import argparse

def get_lines(path, cnt):
    flag=False
    for f in os.listdir(path):        
        if os.path.isdir(f):
            cnt += get_lines(os.path.join(path, f), cnt)
            flag=True
        elif '.py' in f[-3:]:
            with open(os.path.join(path, f), 'r') as f:
                cnt += len(f.readlines())
            flag = True
        else:
            pass
    return cnt if flag else 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser('add some in here')
    parser.add_argument('-dir_path', type=str)
    args = parser.parse_args()
    cnt = 0
    pre = '/media/nile/study/repositorys/autosumma'
    cnt = get_lines(os.path.join(pre, args.dir_path), cnt)
    print(cnt)
