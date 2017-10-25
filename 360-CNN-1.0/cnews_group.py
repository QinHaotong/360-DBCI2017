# -*- coding: utf-8 -*-

"""
将文本整合到 train、test、val 三个文件中
第一次运行前先执行一次这两个函数，生成五个txt文件
"""

import os
import re
import csv

def save_file():
    """
    将多个文件整合并存到3个文件中
    dirname: 原数据目录
    文件内容格式:  类别\t内容
    """
    f_train = open('F:/BDCI2017-360/train.txt', 'w', encoding='utf-8')
    f_test = open('F:/BDCI2017-360/test.txt', 'w', encoding='utf-8')
    f_val = open('F:/BDCI2017-360/val.txt', 'w', encoding='utf-8')

    pCount=0
    nCount=0
    trainfile = "F:/BDCI2017-360/train.tsv"
    with open(trainfile, 'r', encoding='utf-8',newline='') as f:
        lines = [line.strip().split('\t') for line in f.readlines()]
        for line in lines:
            #print(line[0])
            if(line[3]=="POSITIVE"):
                if(pCount<5000):
                    f_test.write(line[3] + '\t' + line[1] + line[2] + '\n')
                elif(pCount<10000):
                    f_val.write(line[3] + '\t' + line[1] + line[2] + '\n')
                else:
                    f_train.write(line[3] + '\t' + line[1] + line[2] + '\n')
                pCount+=1
            elif (line[3]=="NEGATIVE"):
                if(nCount<5000):
                    f_test.write(line[3] + '\t' + line[1] + line[2] + '\n')
                elif(nCount<10000):
                    f_val.write(line[3] + '\t' + line[1] + line[2] + '\n')
                else:
                    f_train.write(line[3] + '\t' + line[1] + line[2] + '\n')
                nCount+=1

    f_train.close()
    f_test.close()
    f_val.close()

def work_file():
    '''
    将输出文件转化为训练集能接受的形式
    分两个文件：evaluation_public.txt 前面加的POSITIVE只是为了复用cnew_loader中的函数，无实际意义
                evaluation_public_id.txt记录各条id ，与上个文件一一对应
    '''
    workfile = 'F:/BDCI2017-360/evaluation_public.tsv'
    f_work = open('F:/BDCI2017-360/evaluation_public.txt', 'w', encoding='utf-8')
    f_id = open('F:/BDCI2017-360/evaluation_public_id.txt', 'w', encoding='utf-8')
    with open(workfile, 'r', encoding='utf-8', newline='') as f:
         for line in f.readlines():
            t=line.strip().split('\t')
            #print(line)
            try:
                f_work.write("POSITIVE" + '\t' + t[1] + t[2] + '\n')
                f_id.write(t[0]+'\n')
            except:
                f_work.write("POSITIVE" + '\t' + t[1] + '\n')
                f_id.write(t[0] + '\n')

if __name__ == '__main__':
    work_file()
    '''
    save_file()
    print(len(open('F:/BDCI2017-360/train.txt', 'r', encoding='utf-8').readlines()))
    print(len(open('F:/BDCI2017-360/test.txt', 'r', encoding='utf-8').readlines()))
    print(len(open('F:/BDCI2017-360/val.txt', 'r', encoding='utf-8').readlines()))
    '''