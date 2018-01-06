import sys
from svmutil import *

class digitRecognition:
    def read_training(self, filename):
        self.yClass = []
        self.xData = []
        for line in open(filename):
            item = line.split(' ')
            self.yClass.append(float(item[0]))
            self.xData.append([float(item[3]), float(item[5].strip('\n'))])

    def read_test(self, filename):
        self.y_test = []
        self.x_test = []
        for line in open(filename):
            item = line.split(' ')
            self.y_test.append(float(item[0]))
            self.x_test.append([float(item[3]), float(item[5].strip('\n'))])

    # 1-versus-all classification with polynomial kernel
    def train(self, digit, C, Q):
        data = []; Y = []
        for i in range(len(self.yClass)):
            data.append(self.xData[i])
            if self.yClass[i] == digit:
                Y.append(1)
            else: Y.append(-1)
        prob = svm_problem(Y, data)
        param = svm_parameter('-c ' + str(C) + ' -t 1 -d ' + str(Q) + ' -r 1 -g 1')
        model = svm_train(prob, param)
        p_label, p_acc, p_val = svm_predict(Y, data, model)

    # 1-versus-1 classification with polynomial kernel
    def train_oneVone(self, d1, d2, C, Q, n):
        data = []; Y = []
        for i in range(len(self.xData)):
            if self.yClass[i] == d1:
                Y.append(1)
                data.append(self.xData[i])
            elif self.yClass[i] == d2:
                Y.append(-1)
                data.append(self.xData[i])
        prob = svm_problem(Y, data)
        param = svm_parameter('-c ' + str(C) + ' -t 1 -d ' + str(Q) + ' -r 1 -g 1 -v ' + str(n) + ' -q')
        model = svm_train(prob, param)
        #pin_label, pin_acc, pin_val = svm_predict(Y, data, model)
        # out-of-sample validation
        xt = []; yt = []
        for j in range(len(self.x_test)):
            if self.y_test[j] == d1:
                yt.append(1)
                xt.append(self.x_test[j])
            elif self.y_test[j] == d2:
                yt.append(-1)
                xt.append(self.x_test[j])
        #p_label, p_acc, p_val = svm_predict(yt, xt, model)
        return model

    # 1-versus-1 classification with RBF kernel
    def train_oneVone_RBF(self, d1, d2, C):
        data = []; Y = []
        for i in range(len(self.xData)):
            if self.yClass[i] == d1:
                Y.append(1)
                data.append(self.xData[i])
            elif self.yClass[i] == d2:
                Y.append(-1)
                data.append(self.xData[i])
        prob = svm_problem(Y, data)
        param = svm_parameter('-c ' + str(C) + ' -t 2 -g 1')
        model = svm_train(prob, param)
        #pin_label, pin_acc, pin_val = svm_predict(Y, data, model)
        xt = []; yt = []
        for j in range(len(self.x_test)):
            if self.y_test[j] == d1:
                yt.append(1)
                xt.append(self.x_test[j])
            elif self.y_test[j] == d2:
                yt.append(-1)
                xt.append(self.x_test[j])
        p_label, p_acc, p_val = svm_predict(yt, xt, model)





dR = digitRecognition()
dR.read_training('train.txt')
dR.read_test('test.txt')
for i in range(-2, 8, 2):
    dR.train_oneVone_RBF(1,5,10**(i))
