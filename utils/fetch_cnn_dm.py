import struct
import json

from tensorflow.core.example import example_pb2


"""
I found the data at https://github.com/abisee/cnn-dailymail
download at https://drive.google.com/file/d/0BzQ6rtO2VN95a0c3TlZCWkl3aU0/view?usp=sharing

but the file is .bin, we need txt
so the file is do this task: transfer .bin file to .txt file
"""


def example_generator(file_path):
    try:
        reader = open(file_path, 'rb')
        while True:
            len_bytes = reader.read(8)
            if not len_bytes: break
            str_len = struct.unpack('q', len_bytes)[0]
            example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
            yield example_pb2.Example.FromString(example_str)
    except Exception as ex:
        print(ex)
    finally:
        reader.close() # in generator, if close() at the finally block, it will excute at final generation


def transferbin2txt(bin_path, txt_path):
    val_lines = []
    for line in example_generator(bin_path):
        content = line.features.feature['article'].bytes_list.value[0].decode('utf-8')
        summary = line.features.feature['abstract'].bytes_list.value[0].decode('utf-8')
        val_lines.append(json.dumps({'content': content, 'summary': summary}, ensure_ascii=False))

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(val_lines))


if __name__ == "__main__":
    transferbin2txt(bin_path='val.bin', txt_path='val.txt')
    transferbin2txt(bin_path='train.bin', txt_path='train.txt')
