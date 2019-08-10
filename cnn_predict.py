# coding: utf-8

from __future__ import print_function
import sys
# sys.path.append(".")
# print(sys.path)

import os
import tensorflow as tf
import tensorflow.contrib.keras as kr
import time
import re

from models.cnn_model import TCNNConfig, TextCNN
from data_processor.dataprocessor import DataProcessor

# try:
#     bool(type(unicode))
# except NameError:
#     unicode = str

class CnnModel:
    def __init__(self):
        self.config = TCNNConfig()
        self.dataprocessor= DataProcessor()

        base_dir = 'dataset'
        base_dic_dir = 'dataDic'
        train_dir = os.path.join(base_dir, 'train')
        test_dir = os.path.join(base_dir, 'test')
        val_dir = os.path.join(base_dir, 'dev')
        vocab_dir = os.path.join(base_dic_dir, 'sms.vocab.txt')

        save_dir = 'checkpoints/textcnn'
        save_path = os.path.join(save_dir, 'best_validation')

        self.config.seq_length = self.dataprocessor.prepareDictory([test_dir,val_dir,train_dir], vocab_dir,0)
        self.categories, self.cat_to_id = self.dataprocessor.read_category()
        self.words, self.word_to_id = self.dataprocessor.read_vocab(vocab_dir)
        self.config.vocab_size = len(self.words)
        self.model = TextCNN(self.config)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)  # 读取保存的模型

    def format_input(self,message):
        self.start_time=time.time()
        message = re.sub(r"(www|http)\S+", "", message)
        print('    remove url:', message)
        segs = self.dataprocessor.pku_seg.cut(message)
        print('    fen ci:',segs)
        list_seg = filter(lambda x: len(x) > 0 and x != '\r\n', segs)
        cut1_list = list(list_seg)
        segs1 = filter(lambda x: x not in self.dataprocessor.stopwords, cut1_list)
        cut2_list = list(segs1)
        print('    remove stops:', cut2_list)
        return cut2_list

    def predict(self, message):
        cut_input = self.format_input(message)
        for i in cut_input:
            if i in self.word_to_id:
                print(i,"  ",self.word_to_id[i])
            else:
                print('outsider of vocabulary',i)
        data = [self.word_to_id[x] for x in cut_input if x in self.word_to_id]
        mode_input = kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length)
        feed_dict = {
            self.model.input_x: mode_input,
            self.model.keep_prob: 1.0
        }

        y_pred_cls = self.session.run(self.model.y_pred_cls, feed_dict=feed_dict)
        print('paint input:',message)
        print('cut input:',cut_input)
        print('mode input:',mode_input)
        print('ouput ->:',y_pred_cls,' <.......',self.categories[y_pred_cls[0]],'.......>')
        print('predict cost:',time.time()-self.start_time)
        return self.categories[y_pred_cls[0]]


if __name__ == '__main__':
    print('predict----..................................')
    cnn_model = CnnModel()
    test_demo = ['【简奕科技】尊敬的会员 :夏文慧您好,您现在已经成功办理开髋主题课卡 卡号为6526',
                 '【优浙点】饭团家·甜品蛋糕披萨生日蛋糕 的顾客已完成评价：6419154608139925']
    for i in test_demo:
        cnn_model.predict(i)
