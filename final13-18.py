from svmutil import *
import numpy as np
import random

def f(x1, x2):
    return np.sign(x2-x1+0.25*np.sin(np.pi*x1))

class RBF:
    def generate(self, N, M):
        self.X = []; self.XX = []
        self.Y = []; self.YY = []
        for i in range(N):
            x1 = random.uniform(-1,1)
            x2 = random.uniform(-1,1)
            y = f(x1,x2)
            self.X.append([x1,x2])
            self.Y.append(y)
        for i in range(M):
             x1 = random.uniform(-1,1)
             x2 = random.uniform(-1,1)
             y = f(x1,x2)
             self.XX.append([x1,x2])
             self.YY.append(y)

    def hardSVM(self, C):
        prob = svm_problem(self.Y, self.X)
        param = svm_parameter('-c ' + str(C) + ' -t 2 -g 1.5 -q')
        model = svm_train(prob, param)
        #pin_label, pin_acc, pin_val = svm_predict(self.Y, self.X, model)
        #return pin_acc[0]
        pout_label, pout_acc, pout_val = svm_predict(self.YY, self.XX, model)
        return pout_acc[0]

    def Lloyd(self, K):
        mu = [[random.uniform(-1,1), random.uniform(-1,1)] for i in range(K)]
        while True:
            cluster = [[] for i in range(K)]
            for x in self.X:
                minDistanceSquared = 8
                for j in range(len(mu)):
                    distSquared = (mu[j][0]-x[0])**2 + (mu[j][1]-x[1])**2
                    if(distSquared < minDistanceSquared):
                        minDistanceSquared = distSquared; place = j
                cluster[place].append(x)
            mu_temp = []
            for i in range(K):
                if len(cluster[i]) == 0:
                    print('Empty Clusters!'); return []
                first = 0; second = 0
                for c in cluster[i]:
                    first += c[0]; second += c[1]
                mu_temp.append([first/len(cluster[i]), second/len(cluster[i])])
            if mu_temp == mu: return mu
            else: mu = mu_temp

    def regularRBF(self, mu, K, gamma):
        Phi = [[np.exp(
               -gamma*((center[0]-x[0])**2 + (center[1]-x[1])**2)) for center in mu]
               for x in self.X]
        for p in Phi:
            p.insert(0,1)
        pseudoInvPhi = np.dot(
                       np.linalg.inv(np.dot(np.transpose(Phi), Phi)),
                       np.transpose(Phi))
        wML = np.dot(pseudoInvPhi, np.transpose(self.Y))
        inError = []
        for i in range(len(self.X)):
            item = [np.exp(
                   -gamma*((center[0]-self.X[i][0])**2 + (center[1]-self.X[i][1])**2)) for center in mu]
            item.insert(0,1)
            if(np.sign(np.dot(wML, item)) == self.Y[i]):
                inError.append(1)
            else: inError.append(-1)
        print('In-sample error for regular RBF', inError.count(-1) / len(inError))
        outError = []
        for i in range(len(self.XX)):
            item = [np.exp(
                   -gamma*((center[0]-self.XX[i][0])**2 + (center[1]-self.XX[i][1])**2)) for center in mu]
            item.insert(0,1)
            if(np.sign(np.dot(wML, item)) == self.YY[i]):
                outError.append(1)
            else: outError.append(-1)
        print('Out-of-sample error for regular RBF', outError.count(-1) / len(outError))
        return outError.count(-1) / len(outError)


Sol = RBF()
total = 0
i = 0
while i < 1000:
    Sol.generate(100,200)
    mu = Sol.Lloyd(12)
    if not mu: continue
    errorRBF = Sol.regularRBF(mu, 12, 1.5)
    errorKernel = 1 - Sol.hardSVM(100000) / 100
    if errorRBF > errorKernel: total += 1
    i += 1
print(total/1000)
