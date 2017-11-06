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
    f_train0 = open('F:/BDCI2017-360/train0.txt', 'w', encoding='utf-8')
    f_train1 = open('F:/BDCI2017-360/train1.txt', 'w', encoding='utf-8')
    f_train2 = open('F:/BDCI2017-360/train2.txt', 'w', encoding='utf-8')
    f_train3 = open('F:/BDCI2017-360/train3.txt', 'w', encoding='utf-8')
    f_train4 = open('F:/BDCI2017-360/train4.txt', 'w', encoding='utf-8')
    f_test = open('F:/BDCI2017-360/test.txt', 'w', encoding='utf-8')
    f_val = open('F:/BDCI2017-360/val.txt', 'w', encoding='utf-8')

    pCount=0
    nCount=0
    count=0
    tCount1 = 0
    tCount2 = 0
    trainfile = "F:/BDCI2017-360/train.tsv"
    with open(trainfile, 'r', encoding='utf-8',newline='') as f:
        lines = [line.strip().split('\t') for line in f.readlines()]
        for line in lines:
            #print(line[0])
            count += 1
            if(line[3]=="POSITIVE"):
                if(pCount%50==0):
                    f_test.write(line[3] + '\t' + line[1] + line[2] + '\n')
                elif(pCount%50==1):
                    f_val.write(line[3] + '\t' + line[1] + line[2] + '\n')
                else:
                    f_train.write(line[3] + '\t' + line[1] + line[2] + '\n')
                    if tCount1%5 == 0:
                        f_train0.write(line[3] + '\t' + line[1] + line[2] + '\n')
                    elif  tCount1%5 == 1:
                        f_train1.write(line[3] + '\t' + line[1] + line[2] + '\n')
                    elif  tCount1%5 == 2:
                        f_train2.write(line[3] + '\t' + line[1] + line[2] + '\n')
                    elif  tCount1%5 == 3:
                        f_train3.write(line[3] + '\t' + line[1] + line[2] + '\n')
                    elif  tCount1%5 == 4:
                        f_train4.write(line[3] + '\t' + line[1] + line[2] + '\n')
                    tCount1+=1
                pCount+=1
            elif (line[3]=="NEGATIVE"):
                if (nCount % 50 == 0):
                    f_test.write(line[3] + '\t' + line[1] + line[2] + '\n')
                elif (nCount % 50 == 1):
                    f_val.write(line[3] + '\t' + line[1] + line[2] + '\n')
                else:
                    f_train.write(line[3] + '\t' + line[1] + line[2] + '\n')
                    if tCount2 % 5 == 0:
                        f_train0.write(line[3] + '\t' + line[1] + line[2] + '\n')
                    elif tCount2 % 5 == 1:
                        f_train1.write(line[3] + '\t' + line[1] + line[2] + '\n')
                    elif tCount2 % 5 == 2:
                        f_train2.write(line[3] + '\t' + line[1] + line[2] + '\n')
                    elif tCount2 % 5 == 3:
                        f_train3.write(line[3] + '\t' + line[1] + line[2] + '\n')
                    elif tCount2 % 5 == 4:
                        f_train4.write(line[3] + '\t' + line[1] + line[2] + '\n')
                    tCount2 += 1
                nCount += 1
        print(pCount,"   ",nCount)

def work_file():
    '''
    将输出文件转化为训练集能接受的形式
    分两个文件：evaluation_public.txt 前面加的POSITIVE只是为了复用cnew_loader中的函数，无实际意义
                evaluation_public_id.txt记录各条id ，与上个文件一一对应
    '''
    count=0
    workfile = 'F:/BDCI2017-360/evaluation_public.tsv'
    f_work0 = open('F:/BDCI2017-360/evaluation_public0.txt', 'w', encoding='utf-8')
    f_work1 = open('F:/BDCI2017-360/evaluation_public1.txt', 'w', encoding='utf-8')
    f_work2 = open('F:/BDCI2017-360/evaluation_public2.txt', 'w', encoding='utf-8')
    f_work3 = open('F:/BDCI2017-360/evaluation_public3.txt', 'w', encoding='utf-8')
    f_id = open('F:/BDCI2017-360/evaluation_public_id.txt', 'w', encoding='utf-8')
    f_id0 = open('F:/BDCI2017-360/evaluation_public_id0.txt', 'w', encoding='utf-8')
    f_id1 = open('F:/BDCI2017-360/evaluation_public_id1.txt', 'w', encoding='utf-8')
    f_id2 = open('F:/BDCI2017-360/evaluation_public_id2.txt', 'w', encoding='utf-8')
    f_id3 = open('F:/BDCI2017-360/evaluation_public_id3.txt', 'w', encoding='utf-8')
    with open(workfile, 'r', encoding='utf-8', newline='') as f:
         for line in f.readlines():
            t=line.strip().split('\t')
            #print(line)
            count+=1
            if count%4==0:
                f_work0.write("POSITIVE" + '\t' + t[1] + t[2] + '\n')
                f_id0.write(t[0] + '\n')
            elif count%4==1:
                f_work1.write("POSITIVE" + '\t' + t[1] + t[2] + '\n')
                f_id1.write(t[0] + '\n')
            elif count%4==2:
                f_work2.write("POSITIVE" + '\t' + t[1] + t[2] + '\n')
                f_id2.write(t[0] + '\n')
            elif count%4==3:
                f_work3.write("POSITIVE" + '\t' + t[1] + t[2] + '\n')
                f_id3.write(t[0]+'\n')
            f_id.write(t[0] + '\n')

if __name__ == '__main__':
    save_file()
    work_file()