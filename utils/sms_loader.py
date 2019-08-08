# coding: utf-8

import sys
from collections import Counter

import numpy as np
import tensorflow.keras as kr
import pandas as pd
import re
import pkuseg
import os

if sys.version_info[0] > 2:
    is_py3 = True
# else:
#     reload(sys)
#     sys.setdefaultencoding("utf-8")
#     is_py3 = False

def native_word(word, encoding='utf-8'):
    """support compatibility between python2 and python3"""
    if not is_py3:
        return word.encode(encoding)
    else:
        return word

def native_content(content):
    if not is_py3:
        return content.decode('utf-8')
    else:
        return content


def open_file(filename, mode='r'):
    """
    support compatibility between python2 and python3
    mode: 'r' or 'w' for read or write
    """
    if is_py3:
        return open(filename, mode, encoding='utf-8', errors='ignore')
    else:
        return open(filename, mode)

def read_file(dir):

    print('read file', dir)
    if not os.path.exists(dir+'/set.csv'):
        print('set.csv no exists')
        g = os.walk(dir)
        paths=[]
        for path,dir_list,file_list in g:
            for file_name in file_list:
                print("filename=",os.path.join(path, file_name))
                paths.append(os.path.join(path, file_name))
        pku_seg = pkuseg.pkuseg(user_dict='dataDic/sms_dic.txt')
        stopwords = pd.read_csv("dataDic/stopwords.txt", index_col=False, quoting=3, sep="\t", names=['stopword'],encoding='utf-8')
        stopwords = stopwords['stopword'].values
        for afile in paths:
            df = pd.read_csv(afile, encoding='utf-8')
            def remove_urls(text):
                text = re.sub(r"(www|http)\S+", "", text)
                return text
            df['CONTENT_removeUrl'] = df['CONTENT'].apply(remove_urls)
            content=df.CONTENT_removeUrl.values.tolist()

        sentences = []
        for line in content:
            segs = pku_seg.cut(line)
            segs = filter(lambda x: len(x) > 1 and x != '\r\n', segs)
            segs = filter(lambda x: x not in stopwords, segs)
            sentences.append((" ".join(segs)))

        df['CLEAN_noUrl_noStop'] = pd.DataFrame({'CLEAN_noUrl_noStop': sentences})

        if 'state!=4' in afile:
            df['LABEL'] = 'rejection'
        elif 'state=4' in afile:
            df['LABEL'] = 'pass'

        df.to_csv(dir + '/set.csv', mode='a')
    else:
        print('set.csv has existed')
        df=pd.read_csv(dir+'/set.csv')

    alldata = []
    for index, content in enumerate(df.CLEAN_noUrl_noStop.values):
        try:
            a = content.split(' ')
            alldata.append(a)
        except:
            print('except:', content, ' ', type(content), " index=", index, " df=", df.iloc[index])
            continue

    return alldata, df.LABEL.to_list()


def build_vocab(train_dir, vocab_dir, vocab_size=5000):

    read_file(train_dir)
    data_train, _ = read_file(train_dir)

    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common()
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')


def read_vocab(vocab_dir):

    with open_file(vocab_dir) as fp:
        # if python2, then convert the value into the unicode format
        words = [native_content(_.strip()) for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def read_category():

    categories = ['rejection','pass']

    categories = [native_content(x) for x in categories]

    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id


def to_words(content, words):
    """transform id presentation into text"""
    return ''.join(words[x] for x in content)


def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """transform text segment unit into id presentation"""
    print('process_file=',filename)
    contents, labels = read_file(filename)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    # keras pad_sequences could service paddding fixed length
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # transform the labels into one-hot presentation

    return x_pad, y_pad


def batch_iter(x, y, batch_size=64):

    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
