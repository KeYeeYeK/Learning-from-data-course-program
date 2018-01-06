import sys
from svmutil import *

class prob17:
    def train(self, C):
        self.X = [[1,0],[0,1],[0,-1],[-1,0],[0,2],[0,-2],[-2,0]]
        self.Y = [-1,-1,-1,1,1,1,1]
        prob = svm_problem(self.Y, self.X)
        param = svm_parameter('-c ' + str(C) + ' -t 1 -d 2 -r 1 -g 1')
        model = svm_train(prob, param)


sol = prob17()
sol.train(10000)
