#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import Counter
import tensorflow.contrib.keras as kr
import numpy as np
import jieba
import os

'''
第一次运行前先看最后一行，去掉注释运行一遍，生成词汇表txt文件
'''


def read_file(filename):
    """读取文件数据"""
    contents = []
    labels = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                contents.append(list(content))
                labels.append(label)
            except:
                pass
    return contents, labels

def build_vocab_last(train_dir, vocab_dir, vocab_size=5000):
    """根据训练集构建词汇表，存储——我忘了这个版本是干啥的了，先当它没用"""
    data_train, _ = read_file(train_dir)
    f=open(vocab_dir, 'w', encoding='utf-8')
    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)

    f.write('\n'.join(words))


def build_vocab(train_dir, vocab_dir, vocab_size=2000000):
    """根据训练集构建词汇表，存储——结巴分词版本，每个词为结巴分词结果（注释部分为非中文字的加入）"""
    f = open(vocab_dir, 'w', encoding='utf-8')
    all_data = []
    temp = []
    count = 0
    with open(train_dir, 'r', encoding='utf-8') as f1:
        for line in f1:
            count+=1
            _, content = line.strip().split('\t')
            seg_list = jieba.cut(content)
            all_data.extend(seg_list)
            if(count%1000==0):
                counter = Counter(all_data)
                all_data = []
                count_pairs = counter.most_common()
                words, _ = list(zip(*count_pairs))
                temp = temp + list(words)
                words = []
                counter = Counter(temp)
                #print(type(counter))
                count_pairs = counter.most_common()
                words, _ = list(zip(*count_pairs))
                temp = list(words)
        counter = Counter(temp)
        count_pairs = counter.most_common(vocab_size - 1)
        words, _ = list(zip(*count_pairs))
        words = ['<PAD>'] + list(words)

        f.write('\n'.join(words))

def build_vocab_last1(train_dir, vocab_dir, vocab_size=2000000):
    """根据训练集构建词汇表，存储——单字版本，每个词为单字"""
    f=open(vocab_dir, 'w', encoding='utf-8')
    all_data = []
    count=0
    A=vocab_size

    with open(train_dir, 'r', encoding='utf-8') as f1:
        letter=[]
        for line in f1:
            label, content = line.strip().split('\t')
            letter.extend(list(content))
            count+=1
            if (count)%10000==0:
                print(count)
                word = []
                for i in range(len(letter) - 1):
                    word.append(letter[i] + letter[i + 1])
                counter = Counter(word)
                count_pairs = counter.most_common(int(vocab_size/20))
                A = A - (int(vocab_size / 20) )
                words, _ = list(zip(*count_pairs))
                if count==10000:
                    # 添加一个 <PAD> 来将所有文本pad为同一长度
                    words = ['<PAD>'] + list(words)
                letter=[]
                word=[]
                f.write('\n'.join(words))

        f1.close()

    with open(train_dir, 'r', encoding='utf-8') as f1:
        letter=[]
        for line in f1:
            label, content = line.strip().split('\t')
            letter.extend(list(content))
            count+=1
            if (count)%10000==0:
                print(count)
                word = []
                for i in range(len(letter) - 2):
                    word.append(letter[i] + letter[i + 1]+ letter[i + 2])
                counter = Counter(word)
                count_pairs = counter.most_common(int(vocab_size/20))
                A = A - (int(vocab_size / 20))
                words, _ = list(zip(*count_pairs))
                letter=[]
                word=[]
                f.write('\n'.join(words))

        f1.close()

    if A>0:
        with open(train_dir, 'r', encoding='utf-8') as f1:
            for line in f1:
                label, content = line.strip().split('\t')
                letter = list(content)
                all_data.extend(letter)
                count += 1
                if (count) % 50000 == 0:
                    print(count)
            counter = Counter(all_data)
            count_pairs = counter.most_common(A-1)
            words, _ = list(zip(*count_pairs))


            f.write('\n'.join(words))

            f1.close()
    elif A<0:
        print("ERROR")

def read_vocab(vocab_dir):
    """读取词汇表"""
    vocab_file = open(vocab_dir, 'r', encoding='utf-8').readlines()
    words = list(map(lambda line: line.strip(),vocab_file))
    word_to_id = dict(zip(words, range(len(words))))

    return words, word_to_id

def read_category():
    """读取分类目录，固定"""
    categories = ['POSITIVE', 'NEGATIVE']
    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id

def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)

def process_file(filename, word_to_id, cat_to_id, max_length=1000):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)

    data_id = []
    label_id = []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id)  # 将标签转换为one-hot表示

    return x_pad, y_pad

def process_file_work(filename, word_to_id, max_length=1000):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)

    data_id = []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)

    return x_pad

def batch_iter(x, y, batch_size=256):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

if __name__ == '__main__':
    build_vocab("F:/BDCI2017-360/train.txt", "F:/BDCI2017-360/vocab.txt", )
    """
    build_vocab("F:/BDCI2017-360/train0.txt", "F:/BDCI2017-360/vocab0.txt", )
    build_vocab("F:/BDCI2017-360/train1.txt", "F:/BDCI2017-360/vocab1.txt", )
    build_vocab("F:/BDCI2017-360/train2.txt", "F:/BDCI2017-360/vocab2.txt", )
    build_vocab("F:/BDCI2017-360/train3.txt", "F:/BDCI2017-360/vocab3.txt", )
    build_vocab("F:/BDCI2017-360/train4.txt", "F:/BDCI2017-360/vocab4.txt", )
    """
