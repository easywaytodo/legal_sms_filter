#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import sys
# sys.path.append(".")
# print(sys.path)
import numpy as np
import os
import tensorflow as tf
import time
from sklearn import metrics
from models.cnn_model import TCNNConfig, TextCNN
from data_processor.dataprocessor import DataProcessor
from datetime import timedelta

base_dir = 'dataset'
base_dic_dir = 'dataDic'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
val_dir = os.path.join(base_dir, 'dev')
vocab_dir = os.path.join(base_dic_dir, 'sms.vocab.txt')

save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict

def evaluate(sess, x_, y_,dataprocessor):

    data_len = len(x_)
    batch_eval = dataprocessor.batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


def train(dataprocessor):
    print("Configuring TensorBoard and Saver...")
    # Before training, please remove tensorboard folder
    os.system('rm -rf tensorboard/textcnn checkpoints/*')
    tensorboard_dir = 'tensorboard/textcnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # config Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Loading training and validation data...")
    # load training set and validation set
    start_time = time.time()
    x_train, y_train = dataprocessor.process_file(train_dir, word_to_id, cat_to_id, config.seq_length)
    x_val, y_val = dataprocessor.process_file(val_dir, word_to_id, cat_to_id, config.seq_length)
    time_dif = get_time_dif(start_time)
    print("process_file  x_train x_val cost-> :", time_dif)

    #  create session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练

    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = dataprocessor.batch_iter(x_train, y_train, config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob)

            if total_batch % config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, x_val, y_val,dataprocessor)

                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0}, Train Loss: {1}, Train Acc: {2},' \
                      + ' Val Loss: {3}, Val Acc: {4}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            feed_dict[model.keep_prob] = config.dropout_keep_prob
            session.run(model.optim, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环
        if flag:  # 同上
            break


def test(dataprocessor):
    print("Loading test data...")
    start_time = time.time()
    x_test, y_test = dataprocessor.process_file(test_dir, word_to_id, cat_to_id, config.seq_length)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # load the model file

    print('Testing...')
    loss_test, acc_test = evaluate(session, x_test, y_test,dataprocessor)
    msg = 'Test Loss: {0}, Test Acc: {1}'
    print(msg.format(loss_test, acc_test))

    batch_size = 128
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_cls = np.argmax(y_test, 1) #get the max on each row
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)  # save the predict results
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)


    # evaluation
    print("111Precision, Recall and F1-Score...")
    print("max-min test.",max(y_test_cls)," ",min(y_test_cls)," pred.",max(y_pred_cls),' ',min(y_pred_cls))
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))
    print('len=',len(y_pred_cls),'   ',len(y_test_cls))
    false_precit=[]
    false_precit_result=[]
    for index,value in enumerate(y_test_cls):
        if y_test_cls[index] != y_pred_cls[index]:
            false_precit.append(index)
            false_precit_result.append([y_test_cls[index],y_pred_cls[index]])
    print('the total number of the wrong prediction:',len(false_precit),' index:',false_precit)
    df=dataprocessor.get_content_by_index(test_dir,false_precit[:5])
    print(false_precit_result[:5])
    df.to_csv('dataset/toImprove.csv')



    # Confusion Matrix.TP FP  /n FN TN
    print("Confusion Matrix:")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    time_dif = get_time_dif(start_time)
    print("test cost time:", time_dif)

if __name__ == '__main__':

    config = TCNNConfig()

    dataprocessor= DataProcessor()

    config.seq_length = dataprocessor.prepareDictory([test_dir,val_dir,train_dir], vocab_dir,0)

    categories, cat_to_id = dataprocessor.read_category()
    words, word_to_id = dataprocessor.read_vocab(vocab_dir)
    config.vocab_size = len(words)
    model = TextCNN(config)
    # train(dataprocessor)
    print('--------------test------------')
    test(dataprocessor)
    # if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
    #     raise ValueError("""usage: python cnn_run.py [train / test]""")

    # if sys.argv[1] == 'train':
    #     train()
    # else:
    #     test()
