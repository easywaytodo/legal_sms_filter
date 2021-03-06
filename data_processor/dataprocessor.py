# coding: utf-8

import sys
from collections import Counter

import numpy as np
import tensorflow.keras as kr
import pandas as pd
import re
import pkuseg
import os
import subprocess
# import sys
# sys.path.append(".")
# print(sys.path)

if sys.version_info[0] > 2:
    is_py3 = True
# else:
#     reload(sys)
#     sys.setdefaultencoding("utf-8")
#     is_py3 = False

class DataProcessor(object):


    def prepareDictory(self, dirs,vocab_dir,trainflag):
        self.pku_seg = pkuseg.pkuseg(user_dict='dataDic/sms_dic.txt')
        fd = open('dataDic/stopwords.txt')
        lines = fd.readlines()
        self.stopwords = [x.strip() for x in lines]
        print('stop:',self.stopwords)

        print('--------------split train,validation,test data set--------------')
        if trainflag == 1:
            pos, neg = self.merge_csv()
            self.build_train_dev_test_set(pos, neg)
            subprocess.getstatusoutput(
                'rm -rf {0}'.format(vocab_dir))


        print('--------------build vocabulary--------------')
        self.max_sequence_length = self.build_vocab(dirs, vocab_dir)


    def build_train_dev_test_set(self, pos, neg):
        file_names=['test','dev','train']
        (status, pos_row_length) = subprocess.getstatusoutput('wc -l ' + pos)
        (status, neg_row_length) = subprocess.getstatusoutput('wc -l ' + neg)
        print(neg_row_length,' ',pos_row_length)
        max_length_of_data_set = min(int(pos_row_length.split(' ')[0]), int(neg_row_length.split(' ')[0]))
        subprocess.getstatusoutput('rm -rf dataset/{0} dataset/{1} dataset/{2}'.format(file_names[0], file_names[1], file_names[2]))
        subprocess.getstatusoutput('mkdir -p dataset/{0} dataset/{1} dataset/{2}'.format(file_names[0],file_names[1],file_names[2]) )
        chunk_size = int(max_length_of_data_set / 5)
        print('chunksize=',chunk_size)
        def saveFiles(file,chunk_size,new_name):
            # pos_rows = pd.read_csv(file, chunksize=chunk_size,error_bad_lines=False)
            pos_rows = pd.read_csv(file, chunksize=chunk_size)
            for i, pos_chuck in enumerate(pos_rows):
                if i > 4:
                    break
                print('chunk', i, ' len=', len(pos_chuck))
                if i< 3:
                    pos_chuck.to_csv('dataset/{0}/{1}'.format(file_names[i], new_name))
                else:
                    pos_chuck.to_csv('dataset/{0}/{1}'.format(file_names[2],new_name),mode='a',header=False)

        saveFiles(pos,chunk_size,'pos_state=4.csv')
        saveFiles(neg, chunk_size,'neg_state!=4.csv')

    def merge_csv(self):

        if not os.path.exists('dataset/pos.csv') or not os.path.exists('dataset/neg.csv'):
            file_names = ['test', 'dev', 'train']
            subprocess.getstatusoutput('rm -rf dataset/{0} dataset/{1} dataset/{2}'.format(file_names[0], file_names[1], file_names[2]))
            g = os.walk('dataset')
            paths = []
            for path, dir_list, file_list in g:
                for file_name in file_list:
                    paths.append(os.path.join(path, file_name))
            print('merge-->',paths)
            rejectionFiles=[]
            passFiles=[]
            for afile in paths:
                if 'state!=4' in afile:
                    rejectionFiles.append(afile)
                elif 'state=4' in afile:
                    passFiles.append(afile)
            print('postive files:',passFiles)
            print('negative files:',rejectionFiles)
            def merge_same_type(file_list, saveFile_Name):
                df = pd.read_csv(file_list[0])
                df.to_csv(saveFile_Name, encoding="utf_8_sig", index=False)
                print('write ',saveFile_Name)
                for i in range(1, len(file_list)):
                    df = pd.read_csv(file_list[i])
                    df.to_csv(saveFile_Name, encoding="utf_8_sig", index=False, header=False, mode='a+')

            merge_same_type(passFiles,'dataset/pos.csv')
            merge_same_type(rejectionFiles, 'dataset/neg.csv')
        return 'dataset/pos.csv','dataset/neg.csv'

    def native_word(self,word, encoding='utf-8'):
        """support compatibility between python2 and python3"""
        if not is_py3:
            return word.encode(encoding)
        else:
            return word

    def native_content(self,content):
        if not is_py3:
            return content.decode('utf-8')
        else:
            return content

    def open_file(self,filename, mode='r'):
        """
        support compatibility between python2 and python3
        mode: 'r' or 'w' for read or write
        """
        if is_py3:
            return open(filename, mode, encoding='utf-8', errors='ignore')
        else:
            return open(filename, mode)

    def read_file(self,dir):

        print('read file', dir)
        if not os.path.exists(dir+'/set.csv'):
            print('set.csv no exists')
            g = os.walk(dir)
            paths=[]
            for path,dir_list,file_list in g:
                for file_name in file_list:
                    print("filename=",os.path.join(path, file_name))
                    paths.append(os.path.join(path, file_name))

            dataframes=[]
            for afile in paths:
                df = pd.read_csv(afile, encoding='utf-8')
                def remove_urls(text):
                    text = re.sub(r"(www|http)\S+", "", text)
                    return text
                df['CONTENT_removeUrl'] = df['CONTENT'].apply(remove_urls)
                content = df.CONTENT_removeUrl.values.tolist()

                sentences = []
                for line in content:
                    segs = self.pku_seg.cut(line)
                    segs = filter(lambda x: len(x) > 1 and x != '\r\n', segs)
                    cut1_list = list(segs)
                    segs1 = filter(lambda x: x not in self.stopwords, cut1_list)
                    cut2_list = list(segs1)
                    sentences.append((" ".join(cut2_list)))

                df['CLEAN_noUrl_noStop'] = pd.DataFrame({'CLEAN_noUrl_noStop': sentences})

                if 'state!=4' in afile:
                    df['LABEL'] = 'rejection'
                elif 'state=4' in afile:
                    df['LABEL'] = 'pass'

                dataframes.append(df)
            result=pd.concat(dataframes,ignore_index=True,axis=0)
            result.to_csv(dir+'/set.csv',index=False)
        else:
            print('set.csv has existed')
            df=pd.read_csv(dir+'/set.csv')

        alldata = []
        for index, content in enumerate(df.CLEAN_noUrl_noStop.values):
            try:
                a = content.split(' ')
                alldata.append(a)
            except:
                print('except: content=', content, ',type=', type(content), " index=", index, " df=", df.iloc[index])
                continue

        return alldata, df.LABEL.to_list()


    def build_vocab(self,dirs, vocab_dir):

        dataset=[]
        for dir in dirs:
            data_content, _ = self.read_file(dir)
            dataset.extend(data_content)


        maxSequence = max(dataset, key=len)
        print(' max length:',len(maxSequence)," ",maxSequence)

        if not os.path.exists(vocab_dir):
            all_data = []
            for content in dataset:
                all_data.extend(content)

            counter = Counter(all_data)
            count_pairs = counter.most_common()
            print('vocabulary length=',len(count_pairs))
            words, _ = list(zip(*count_pairs))
            words = ['<PAD>'] + list(words)
            self.open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')

        return len(maxSequence)

    def read_vocab(self,vocab_dir):

        with self.open_file(vocab_dir) as fp:
            # if python2, then convert the value into the unicode format
            words = [self.native_content(_.strip()) for _ in fp.readlines()]
        word_to_id = dict(zip(words, range(len(words))))
        return words, word_to_id


    def read_category(self,):

        categories = ['rejection','pass']
        categories = [self.native_content(x) for x in categories]
        cat_to_id = dict(zip(categories, range(len(categories))))

        return categories, cat_to_id

    def to_words(self,content, words):
        """transform id presentation into text"""
        return ''.join(words[x] for x in content)


    def process_file(self,filename, word_to_id, cat_to_id, max_length=600):
        """transform text segment unit into id presentation"""
        print('process_file=',filename)
        contents, labels = self.read_file(filename)

        data_id, label_id = [], []
        for i in range(len(contents)):
            data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
            label_id.append(cat_to_id[labels[i]])

        # keras pad_sequences could service paddding fixed length
        x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
        y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # transform the labels into one-hot presentation

        return x_pad, y_pad


    def batch_iter(self,x, y, batch_size=64):

        data_len = len(x)
        num_batch = int((data_len - 1) / batch_size) + 1

        indices = np.random.permutation(np.arange(data_len))
        x_shuffle = x[indices]
        y_shuffle = y[indices]

        for i in range(num_batch):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len)
            yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
