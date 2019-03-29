from unidecode import unidecode

L = []

with open('non_ascii_chars.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        c = line[2]
        L.append(c + ' ' + unidecode(c))

with open('replace_dict.txt', 'w', encoding='utf-8') as f:
    f.writelines('\n'.join(L))
