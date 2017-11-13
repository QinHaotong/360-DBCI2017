#!/usr/bin/python
# -*- coding: utf-8 -*-

from cnn_model import *
from cnews_loader import *
from sklearn import metrics
import sys

import time
from datetime import timedelta

"""
运行方式：python xxxx.py train (参数可换为 test/work)
"""

base_dir = 'F:/BDCI2017-360/'
train_dir = os.path.join(base_dir, 'train.txt')
test_dir = os.path.join(base_dir, 'test.txt')
val_dir = os.path.join(base_dir, 'val.txt')
vocab_dir = os.path.join(base_dir, 'vocab.txt')


save_dir = 'F:/BDCI2017-360/'
save_path = os.path.join(save_dir, 'best_validation.txt')   # 最佳验证结果保存路径

BATCH_SIZE = 256

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

def evaluate(sess, x_, y_):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, BATCH_SIZE)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len

def retrain():
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    tensorboard_dir = 'tensorboard/textcnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    # 载入训练集与验证集
    start_time = time.time()

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 创建session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)



    train_DIR=['F:/BDCI2017-360/train0.txt','F:/BDCI2017-360/train1.txt','F:/BDCI2017-360/train2.txt','F:/BDCI2017-360/train3.txt','F:/BDCI2017-360/train4.txt']
    vocab_DIR=['F:/BDCI2017-360/vocab0.txt','F:/BDCI2017-360/vocab1.txt','F:/BDCI2017-360/vocab2.txt','F:/BDCI2017-360/vocab3.txt','F:/BDCI2017-360/vocab4.txt']
    train_num=5

    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练

    for i in range(train_num):
        print('Training and evaluating...')
        start_time = time.time()
        last_improved = total_batch  # 记录上一次提升批次
        print("Loading training and validation data...")
        words, word_to_id = read_vocab(vocab_dir)
        config.vocab_size = len(words)
        x_train, y_train = process_file(train_DIR[i], word_to_id, cat_to_id, config.seq_length)
        x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, config.seq_length)
        flag = False
        for epoch in range(config.num_epochs):
            print('Epoch:', epoch + 1)
            batch_train = batch_iter(x_train, y_train, config.batch_size)
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
                    loss_val, acc_val = evaluate(session, x_val, y_val)  # todo

                    if acc_val > best_acc_val:
                        # 保存最好结果
                        best_acc_val = acc_val
                        last_improved = total_batch
                        saver.save(sess=session, save_path=save_path)
                        improved_str = '*'
                    else:
                        improved_str = ''

                    time_dif = get_time_dif(start_time)
                    msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                        + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                    print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

                session.run(model.optim, feed_dict=feed_dict)  # 运行优化
                total_batch += 1

                if total_batch - last_improved > require_improvement:
                    # 验证集正确率长期不提升，提前结束训练
                    print("No optimization for a long time, auto-stopping...")
                    flag = True
                    break  # 跳出循环
            if flag:  # 同上
                break


def train():
    if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
        build_vocab(train_dir, vocab_dir, config.vocab_size)
    words, word_to_id = read_vocab(vocab_dir)
    config.vocab_size = len(words)
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    tensorboard_dir = 'tensorboard/textcnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Loading training and validation data...")
    # 载入训练集与验证集
    start_time = time.time()
    x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, config.seq_length)
    x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, config.seq_length)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 创建session
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
        batch_train = batch_iter(x_train, y_train, config.batch_size)
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
                loss_val, acc_val = evaluate(session, x_val, y_val)  # todo

                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            session.run(model.optim, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环
        if flag:  # 同上
            break


def test():
    '''
    测试集测试，这个测试集是一开始分好的test.txt，不参与训练，基本跟最终分数正相关
    '''
    print("Loading test data...")
    words, word_to_id = read_vocab(vocab_dir)
    config.vocab_size = len(words)
    start_time = time.time()
    x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, config.seq_length)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    print('Testing...')
    loss_test, acc_test = evaluate(session, x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = BATCH_SIZE
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)  # 保存预测结果
    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)

    # 评估
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def work(dir1,dir2,dir3):
    '''
    运行之后生成提交文件submit1.csv
    '''
    print("Loading work data...")
    words, word_to_id = read_vocab(vocab_dir)
    config.vocab_size = len(words)
    # start_time = time.time()
    work_dir = os.path.join(base_dir, dir1)#'evaluation_public3.txt'
    x_test = process_file_work(work_dir, word_to_id, config.seq_length)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    print('Testing...')
    # loss_test, acc_test = evaluate(session, x_test, y_test)
    # msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    # print(msg.format(loss_test, acc_test))

    batch_size = BATCH_SIZE
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    # y_test_cls = np.argmax(y_test, 1)
    print(len(x_test))
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)  # 保存预测结果
    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)
    count = 0
    f_t1 = open(dir2, 'w', encoding='utf-8')#"F:/BDCI2017-360/submit3.csv"
    # f_t2 = open("F:/BDCI2017-360/submit2.csv", 'w', encoding='utf-8')
    with open(dir3, 'r', encoding='utf-8') as f1:#"F:/BDCI2017-360/evaluation_public_id3.txt"
        for line in f1:
            line1 = line.replace("\n", "")
            if (y_pred_cls[count] == 0):
                f_t1.write(line1 + ',' + "POSITIVE" + ',' + '\n')
            elif (y_pred_cls[count] == 1):
                f_t1.write(line1 + ',' + "NEGATIVE" + ',' + '\n')
            count += 1

    print(count)


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test', 'work', 'retrain']:
        raise ValueError("""usage: python run_cnn.py [train / test]""")

    print('Configuring CNN model...')
    config = TCNNConfig()
    categories, cat_to_id = read_category()
    # words, word_to_id = read_vocab(vocab_dir)
    # config.vocab_size = len(words)
    model = TextCNN(config)

    if sys.argv[1] == 'train':
        train()
    elif sys.argv[1] == 'retrain':
        retrain()
        work("evaluation_public0.txt", "F:/BDCI2017-360/submit0.csv", "F:/BDCI2017-360/evaluation_public_id0.txt")
        work("evaluation_public1.txt", "F:/BDCI2017-360/submit1.csv", "F:/BDCI2017-360/evaluation_public_id1.txt")
        work("evaluation_public2.txt", "F:/BDCI2017-360/submit2.csv", "F:/BDCI2017-360/evaluation_public_id2.txt")
        work("evaluation_public3.txt", "F:/BDCI2017-360/submit3.csv", "F:/BDCI2017-360/evaluation_public_id3.txt")
    elif sys.argv[1] == 'work':
        pass
    else:
        test()
