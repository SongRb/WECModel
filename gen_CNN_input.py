# This script runs on linux guest machine only
import word2vec
import numpy as np
import pickle

import scipy.spatial.distance as dis

word_embedding_db_path = '/home/tang/Documents/word2vec-python/yahoo-text.bin'
model = word2vec.load(word_embedding_db_path)

trans_matrix = np.load('t_matrix.npy')


n_f = 50
m_f = 100

def generate_pair():
    QA_pair = list()
    with open("book") as textfile1, open("bookTag") as textfile2: 
        for ques_list, ans_list in zip(textfile1, textfile2):
            ques_list = ques_list.strip().split(' ')
            ans_list = ans_list.strip().split(' ')
            if len(ques_list)<n_f and len(ans_list)<m_f:
                QA_pair.append((ques_list,ans_list))

        with open('qa_pair.pkl','wb') as fout:
            pickle.dump(QA_pair,fout)

def load_pair():
    with open('qa_pair.pkl','rb') as fin:
        return pickle.load(fin)

def get_value(word):
    try:
        return model[word]
    except KeyError:
        return np.random.uniform(low=-0.1,high=0.1,size=100)

def cal_matrix(qa_pair):
    total_db = list()
    total_length = len(qa_pair)
    count=0
    for p in qa_pair:
        count+=1
        if count%100 == 0:
            print('{0} in {1} pairs'.format(count,total_length))
        matrix = np.zeros((n_f,m_f))
        for i in range(n_f):
            for j in range(m_f):
                cost = dis.cosine(get_value(p[0][i%len(p[0])]),np.matmul(get_value(p[1][j%len(p[1])]), trans_matrix))
                matrix[i,j] = cost
        total_db.append(matrix)
    return total_db

res = cal_matrix(load_pair())

import pickle
try:
    with open('res.pkl','wb') as fout:
        pickle.dump(res,fout)
except:
    print('weee')
