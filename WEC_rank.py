from __future__ import print_function
import os
import random

import numpy as np
import scipy.spatial.distance as dis
import word2vec
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from parse_text import remove_punctuation

def increment(counter,key):
    if key in counter:
        counter[key]+=1
    else: 
        counter[key]=1

def create_point(counter):
    # key: length
    # value: numbers
    x_1 = list()
    y_1 = list()
    for k,v in counter.iteritems():
        x_1.append(int(k))
        y_1.append(int(v))
    return x_1,y_1

def draw_graph(x,y,name):
    fig = plt.figure()
    
    plt.plot(x, y, 'b,')
    # plt.axis([0, 4000, 0, 1000])
    plt.xticks(np.arange(min(x), max(x)+1, (max(x)-min(x))/20))
    ax = fig.axes[0]
    # ax.autoscale(enable=True)
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    # plt.show()
    fig.suptitle(' '.join(name.split('_')).title())
    fig.savefig('{0}.svg'.format(name),figsize=(2000,2000),dpi=1000)
    # print rate_len_counter
    
    


class WECRanker:
    def __init__(self):
        work_dir = os.path.join('work', 'L6-1')
        WORD2VEC_MODEL_PATH = os.path.join(work_dir, 'YahooTextL6-1.bin')
        WEC_MODEL_PATH = os.path.join(
            work_dir, '20171031-141035-13200', 't_matrix.npy')

        self.w2v_model = word2vec.load(WORD2VEC_MODEL_PATH)
        self.t_matrix = np.load(WEC_MODEL_PATH)

        with open(os.path.join(work_dir, 'answer'), 'r') as ans_in, open(os.path.join(work_dir, 'question'),
                                                                         'r') as ques_in:
            self.answer_list = ans_in.read().split('\n')
            self.question_list = ques_in.read().split('\n')
            self.answer_list = [unicode(i) for i in self.answer_list]
            self.question_list = [unicode(i) for i in self.question_list]

        ans_len_counter = dict()
        ques_len_counter = dict()
        rate_len_counter = dict()
        for index in xrange(len(self.answer_list)):
            ans_length = len(self.answer_list[index])
            if ans_length == 0:
                continue 
            ques_length = len(self.question_list[index])
            increment(ans_len_counter,ans_length)
            increment(ques_len_counter,ques_length)
            try:
                increment(rate_len_counter,float(ans_length)/ques_length)
            except ZeroDivisionError:
                pass
        # for i in xrange(100):
        #     if i in ans_len_counter:
        #         print(i,': ',ans_len_counter[i])
        x_1,y_1 =create_point(ans_len_counter)
        x_2,y_2 = create_point(ques_len_counter)
        x_3,y_3 = create_point(ques_len_counter)

        draw_graph(x_1,y_1,'answer_length')
        draw_graph(x_2,y_2,'question_length')
        draw_graph(x_3,y_3,'a2q_length_ratio')


    @staticmethod
    def process_text(s):
        s = remove_punctuation(s.lower())
        return s.strip().split(' ')

    def cal_score(self, answer, question):
        answer_words = self.process_text(answer)
        question_words = self.process_text(question)

        total_score = list()
        for i in answer_words:
            max_score = 0
            if i in self.w2v_model:
                eli_ans_vec = np.matmul(self.t_matrix, self.w2v_model[i])
                for j in question_words:
                    if j in self.w2v_model:
                        score = 1 - dis.cosine(self.w2v_model[j], eli_ans_vec)
                        if score > max_score:
                            max_score = score
                total_score.append(max_score)
        if len(total_score) != 0:
            return sum(total_score) / len(total_score)
        return 0

    def cal_score_list(self, answer_list, question):
        return [self.cal_score(answer, question) for answer in answer_list]

    def validate_model(self):
        count = 0
        for index in range(len(self.question_list)):
            if index % 50 == 0:
                print('Rate:', float(count) / (index + 1),"{0} processed".format(index))
            qa_pair = (self.question_list[index], self.answer_list[index])
            answer_list = random.sample(self.question_list, 5)
            score_list = self.cal_score_list(answer_list, qa_pair[0])
            score = self.cal_score(qa_pair[1], qa_pair[0])
            score_list = sorted(score_list)
            if score > score_list[-1]:
                count += 1

        return float(count) / len(self.question_list)


if __name__ == '__main__':
    import sys

    reload(sys)
    sys.setdefaultencoding("utf-8")

    ranker = WECRanker()
    print(ranker.validate_model())
