# 学校:北京航空航天大学
# 姓名:李文雯
# 学号:ZY2203106
# 日期:2023.3.25


import numpy as np
import os
import re
import math
import time
import jieba
DATA_PATH = '../jyxstxtqj/'
path = './cn_stopwords.txt'
def calculate_entropy1(words_tf, length_data):#计算一元词的信息熵
    t1 = time.time()
    words_num = sum([item[1] for item in words_tf.items()])

    print("分词种类数：{}".format(words_num))
    print('不同词个数：{}'.format((len(words_tf))))
    print("平均词长：{:.4f}".format(length_data/float(words_num)))

    entropy = 0
    for item in words_tf.items():
        entropy += -(item[1]/words_num) * math.log(item[1]/words_num, 2)
    print("基于分词的一元模型中文信息熵为：{:.4f} 比特/词".format(entropy))

    print("一元模型运行时间：{:.4f} s".format(time.time() - t1))
    return ['unigram model', length_data, len(words_tf), round(length_data/float(words_num), 4), round(entropy, 4)]

def calculate_entropy2(words_tf, bigram_tf, length_data):#计算二元词的信息熵
    t1 = time.time()
    bi_words_num = sum([item[1] for item in bigram_tf.items()])
    avg_word_length = sum(len(item[0][i]) for item in bigram_tf.items() for i in range(len(item[0]))) / len(bigram_tf)

    print("分词种类数：{}".format(bi_words_num))
    print('不同词个数：{}'.format((len(bigram_tf))))
    print("平均词长：{:.4f}".format(avg_word_length))

    entropy = 0
    for bi_item in bigram_tf.items():
        jp = bi_item[1] / bi_words_num
        cp = bi_item[1] / words_tf[bi_item[0][0]]
        entropy += -jp * math.log(cp, 2)
    print("基于分词的二元模型中文信息熵为：{:.4f} 比特/词".format(entropy))

    print("二元模型运行时间：{:.4f} s".format(time.time() - t1))
    return ['bigram model', length_data, len(bigram_tf), round(avg_word_length, 4), round(entropy, 4)]

def calculate_entropy3(bigram_tf, trigram_tf, length_data):#计算三元词的信息熵

    t1 = time.time()
    tri_words_num = sum([item[1] for item in trigram_tf.items()])
    avg_word_length = sum(len(item[0][i]) for item in trigram_tf.items() for i in range(len(item[0])))/len(trigram_tf)

    print("分词种类数：{}".format(tri_words_num))
    print('不同词个数：{}'.format((len(trigram_tf))))
    print("平均词长：{:.4f}".format(avg_word_length))

    entropy = 0
    for tri_item in trigram_tf.items():
        jp = tri_item[1] / tri_words_num
        cp = tri_item[1] / bigram_tf[tri_item[0][0]]
        entropy += -jp * math.log(cp, 2)
    print("基于分词的三元模型中文信息熵为：{:.4f} 比特/词".format(entropy))

    print("三元模型运行时间：{:.4f} s".format(time.time() - t1))
    return ['trigram model', length_data, len(trigram_tf), round(avg_word_length, 4), round(entropy, 4)]



def read_data(file_path,path):#获取file_path文件对应的内容

    corpus = ''
    r = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:：;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~「」『』（）]+'
    with open(file_path, 'r', encoding='ANSI') as f:
        corpus = f.read()
        corpus = re.sub(r, '', corpus)
        corpus = corpus.replace('\n', '')
        corpus = corpus.replace('\u3000', '')
        corpus = corpus.replace('本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com', '')
        f.close()
    with open(path, 'r', encoding='UTF-8') as f:
        stopwords = []
        for a in f:
            if a != '\n':
                stopwords.append(a.strip())
    for a in stopwords:
        corpus = corpus.replace(a, '')
    return corpus


def all_corpus(index_file):#获取所有语料库文件的内容列表

    index_path = index_file
    whole_corpus = []
    with open(index_path, 'r') as f:
        txt_list = f.readline().split(',')
        print("要求解信息熵的文件列表为：")
        for file in txt_list:
            print(file)
            file_path = DATA_PATH + file + '.txt'
            whole_corpus.append(read_data(file_path,path))
        print('---------------------------')
        f.close()
    return ''.join(whole_corpus)

def word_tf(tf1, words):#获取一元词词频

    for i in range(len(words)):
        tf1[words[i]] = tf1.get(words[i], 0) + 1

def bigram_tf(tf2, words):#获取二元词词频

    for i in range(len(words)-1):
        tf2[(words[i], words[i+1])] = tf2.get((words[i], words[i+1]), 0) + 1

def trigram_tf(tf3, words):#获取三元词词频

    for i in range(len(words)-2):
        tf3[((words[i], words[i+1]), words[i+2])] = tf3.get(((words[i], words[i+1]), words[i+2]), 0) + 1

def print_md(table_name, head, row_title, col_title, data):
    """
    table_name: 表名 head: 表头 row_title: 行名，编号，1，2，3…… col_title: 列名，词数，运行时间等 data: {ndarray(H, W)}
    """
    element = " {} |"

    h, w = len(data), len(data[0])
    lines = ['#### {}'.format(table_name)]

    lines += ["| {} | {} |".format(head, ' | '.join(col_title))]

    # 分割线
    split = "{}:{}"
    line = "| {} |".format(split.format('-' * len(head), '-' * len(head)))
    for i in range(w):
        line = "{} {} |".format(line, split.format('-' * len(col_title[i]), '-' * len(col_title[i])))
    lines += [line]

    # 数据部分
    for i in range(h):
        d = list(map(str, list(data[i])))
        lines += ["| {} | {} |".format(row_title[i], ' | '.join(d))]

    table = '\n'.join(lines)
    print(table)
    return table


def calculate_entropy_all(inf, mode):#计算一系列文件的信息熵
    tf1 = {}
    tf2 = {}
    tf3 = {}
    split_words = []
    data = all_corpus(inf)

    print("语料库字数：{}".format(len(data)))
    if mode == 'token':
        split_words = list(jieba.cut(data))
        words_num = len(split_words)
    elif mode == 'char':
        split_words = [ch for ch in data]
    word_tf(tf1, split_words)
    bigram_tf(tf2, split_words)
    trigram_tf(tf3, split_words)
    rows = []
    print('---------------------------------')
    rows.append(calculate_entropy1(tf1, len(data)))
    print('---------------------------------')
    rows.append(calculate_entropy2(tf1, tf2, len(data)))
    print('---------------------------------')
    rows.append(calculate_entropy3(tf2, tf3, len(data)))
    print('---------------------------------')

    head = "#"
    row_title = [str(i + 1) for i in range(len(rows))]
    col_title = ['分词模型', '语料字数', '分词种类数', '平均词长', '信息熵']
    print_md('金庸小说全集信息熵表', head, row_title, col_title, rows)


def calculate_entropy_every(mode):#计算DATA_PATH下每个文件单独的信息熵
    tf1 = {}
    tf2 = {}
    tf3 = {}
    split_words = []
    rows = []

    for file in os.listdir(DATA_PATH):
        print("\n当前计算信息熵的文件为：{}".format(file))
        file_path = DATA_PATH + file

        data = read_data(file_path,path)
        print("语料库字数：{}".format(len(data)))
        if mode == 'token':
            split_words = list(jieba.cut(data))
            words_num = len(split_words)
        elif mode == 'char':
            split_words = [ch for ch in data]
        word_tf(tf1, split_words)
        bigram_tf(tf2, split_words)
        trigram_tf(tf3, split_words)
        rows1 = []
        print('---------------------------------')
        rows1.append(calculate_entropy1(tf1, len(data)))
        print('---------------------------------')
        rows1.append(calculate_entropy2(tf1, tf2, len(data)))
        print('---------------------------------')
        rows1.append(calculate_entropy3(tf2, tf3, len(data)))
        print('---------------------------------')


        rows.append([file.split('.')[0], rows1[0][1], rows1[0][2], rows1[0][3], rows1[0][4],
                     rows1[1][2], rows1[1][3], rows1[1][4],
                     rows1[2][2], rows1[2][3], rows1[2][4]])
        avg_entropy = np.mean([rows1[0][4], rows1[1][4], rows1[2][4]])  # 平均信息熵
        print('平均信息熵： %.4f' % avg_entropy)
        print('---------------------------------')

        #rows.append(round(avg_entropy, 4))
    head = '#'
    row_title = [str(i + 1) for i in range(len(rows))]
    col_title = ['小说名', '语料字数', '一元分词个数', '一元平均词长', '一元模型信息熵', '二元分词个数', '二元平均词长', '二元模型信息熵',
                 '三元分词个数', '三元平均词长', '三元模型信息熵']
    print_md('金庸小说单本信息熵表', head, row_title, col_title, rows)

def calculate_result():#按字或词计算金庸小说全集和单集的信息熵
    calculate_entropy_all('../inf.txt', mode='token')
    calculate_entropy_all('../inf.txt', mode='char')
    calculate_entropy_every(mode='token')
    calculate_entropy_every(mode='char')

if __name__ == "__main__":
    calculate_result()


