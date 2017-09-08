import os
import random

import numpy as np
import scipy.spatial.distance as dis
import word2vec

from parse_text import remove_punctuation


class WECRanker:
    def __init__(self):
        WORD2VEC_MODEL_PATH = os.path.join('data', 'yahoo-text.bin')
        WEC_MODEL_PATH = os.path.join('data', 't_matrix.npy')

        self.w2v_model = word2vec.load(WORD2VEC_MODEL_PATH)
        self.t_matrix = np.load(WEC_MODEL_PATH)

        with open(os.path.join('data', 'answer'), 'r') as ans_in, open(os.path.join('data', 'question'),
                                                                       'r') as ques_in:
            self.answer_list = ans_in.read().split('\n')
            self.question_list = ques_in.read().split('\n')
            self.answer_list = [unicode(i) for i in self.answer_list]
            self.question_list = [unicode(i) for i in self.question_list]

        pass

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
                for j in question_words:
                    if j in self.w2v_model:
                        score = dis.cosine(self.w2v_model[j], np.matmul(self.w2v_model[i], self.t_matrix))
                        if score > max_score:
                            max_score = score
            total_score.append(max_score)

        return sum(total_score) / len(total_score)

    def cal_set(self, answer_list, question):
        return [self.cal_score(answer, question) for answer in answer_list]

    def validate_model(self):
        count = 0
        for index in range(len(self.question_list)):
            if index%200 == 0:
                print 'Rate:',float(count)/(index+1),
                print "{0} processed".format(index)
            qa_pair = (self.question_list[index], self.answer_list[index])
            answer_list = random.sample(self.question_list, 50)
            score_list = self.cal_set(answer_list, qa_pair[0])
            score = self.cal_score(qa_pair[1], qa_pair[0])
            if score >= sorted(score_list)[-2]:
                count += 1

        return float(count) / len(self.question_list)


if __name__ == '__main__':
    import sys

    reload(sys)
    sys.setdefaultencoding("utf-8")

    ranker = WECRanker()
    print ranker.validate_model()
