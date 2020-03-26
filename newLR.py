import math
import datetime
import sys
import numpy as np


class LR:
    def __init__(self, train_file_name, test_file_name, predict_result_file_name):
        self.train_file = train_file_name
        self.predict_file = test_file_name
        self.predict_result_file = predict_result_file_name
        self.max_iters = 200
        self.rate = 0.1#
        self.feats = []
        self.labels = []
        self.feats_test = []
        self.labels_predict = []
        self.param_num = 0
        self.weight = []
        self.lr=2e-3#
        self.b=0#
        self.eps=5e-4

    def loadDataSet(self, file_name, label_existed_flag):
        feats = []
        labels = []
        fr = open(file_name)
        lines = fr.readlines()
        for line in lines:
            temp = []
            allInfo = line.strip().split(',')
            dims = len(allInfo)
            if label_existed_flag == 1:
                for index in range(dims-1):
                    temp.append(float(allInfo[index]))
                feats.append(temp)
                labels.append(float(allInfo[dims-1]))
            else:
                for index in range(dims):
                    temp.append(float(allInfo[index]))
                feats.append(temp)
        fr.close()
        feats = np.array(feats)
        labels = np.array(labels)
        return feats, labels

    def loadTrainData(self):
        self.feats, self.labels = self.loadDataSet(self.train_file, 1)

    def loadTestData(self):
        self.feats_test, self.labels_predict = self.loadDataSet(
            self.predict_file, 0)

    def savePredictResult(self):
        print(self.labels_predict)
        f = open(self.predict_result_file, 'w')
        for i in range(len(self.labels_predict)):
            f.write(str(self.labels_predict[i])+"\n")
        f.close()

    def sigmod(self, x):
        return 1/(1+np.exp(-x))

    def printInfo(self):
        print(self.train_file)
        print(self.predict_file)
        print(self.predict_result_file)
        print(self.feats)
        print(self.labels)
        print(self.feats_test)
        print(self.labels_predict)

    def initParams(self):
        self.weight = np.ones((self.param_num,), dtype=np.float)

    def compute(self, recNum, param_num, feats, w):
        return self.sigmod(np.dot(feats, w)+self.b)

    def error_rate(self, recNum, label, preval):
        return np.power(label - preval, 2).sum()

    def predict(self):
        self.loadTestData()
        self.labels_predict = (self.sigmod(np.dot(self.feats_test, self.weight) + self.b)+0.5).astype(np.int)#阈值判定，int是只取整数部分
        self.savePredictResult()

    def train(self):
        pre_loss=float("inf")
        self.loadTrainData()
        recNum = len(self.feats)
        self.param_num = len(self.feats[0])
        self.initParams()
        ISOTIMEFORMAT = '%Y-%m-%d %H:%M:%S,f'
        for i in range(self.max_iters):
            for x, y in zip(self.feats, self.labels):
                gradient = y - self.sigmod(np.dot(x, self.weight) + self.b)
                self.b += self.lr * gradient
                self.weight += (self.lr * gradient * x).reshape(-1)
            A = np.dot(self.feats, self.weight) + self.b
            loss = (self.labels-A).mean()       # 计算平均损失
            if abs(pre_loss - loss) < self.eps:    # 上一次损失减去当前损失小于一定阈值则停止迭代
                print("finish:"+str(i))
                break
            pre_loss = loss


def print_help_and_exit():
    print("usage:python3 main.py train_data.txt test_data.txt predict.txt [debug]")
    sys.exit(-1)


def parse_args():
    debug = False
    if len(sys.argv) == 2:
        if sys.argv[1] == 'debug':
            print("test mode")
            debug = True
        else:
            print_help_and_exit()
    return debug


if __name__ == "__main__":
    #debug = parse_args()
    train_file =  "./data/train_data.txt"
    test_file = "./data/test_data.txt"
    predict_file = "./data/result.txt"
    lr = LR(train_file, test_file, predict_file)
    lr.train()
    lr.predict()
    debug=True

    if debug:
        answer_file ="./data/answer.txt"
        f_a = open(answer_file, 'r')
        f_p = open(predict_file, 'r')
        a = []
        p = []
        lines = f_a.readlines()
        for line in lines:
            a.append(int(float(line.strip())))
        f_a.close()

        lines = f_p.readlines()
        for line in lines:
            p.append(int(float(line.strip())))
        f_p.close()

        print("answer lines:%d" % (len(a)))
        print("predict lines:%d" % (len(p)))

        errline = 0
        for i in range(len(a)):
            if a[i] != p[i]:
                errline += 1

        accuracy = (len(a)-errline)/len(a)
        print("accuracy:%f" %(accuracy))
