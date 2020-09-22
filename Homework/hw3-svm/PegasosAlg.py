from util import *
import numpy as np
import pickle
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd

def load_data(flnm):
    infile = open(flnm,'rb')
    p_data = pickle.load(infile)
    infile.close()
    return p_data
class pegasos():
    def __init__(self, reg_l, max_epoch):
        self.reg_l = reg_l
        self.max_epoch = max_epoch
    def fit(self, X, y=None):
        num_pnts = len(X_train)
        t = 0
        s = 1
        W = Counter()
        epoch = 0
        while epoch<self.max_epoch:
            epoch+=1
            for j in range(num_pnts):
                print ('t={}'.format(t),end='\r')
                t+=1
                eta_t = 1/(t*self.reg_l)
                if y[j]*dotProduct(W,X[j])<1/s:
                    s*=(1-eta_t*self.reg_l)
                    if s == 0:
                        s = 1
                        W = Counter()
                        continue
                    increment(W,eta_t*y[j]/s,X[j])
                else:
                    s*=(1-eta_t*self.reg_l)
                    if s == 0:
                        s = 1
                        W = Counter()
                        continue
        for f,v in W.items():
            W[f] = v*s
        self.w_ = W
        return self
    def score(self,X,y=None):
        num_pnts = len(X)
        error=0
        for j in range(num_pnts):
            if y[j]*dotProduct(self.w_,X[j]) >0:
                pass
            else:
                error+=1
        return error/num_pnts
    def predict(self,X,y=None):
        num_pnts = len(X)
        predicts = []
        correct = []
        for j in range(num_pnts):
            predicts.append(dotProduct(self.w_,X[j]))
            if y[j]*dotProduct(self.w_,X[j]) >0:
                correct.append(1)
            else:
                correct.append(0)
        return predicts,correct


#    def fit2(self, X, y=None):
#        print ('start fitting, method 1 ')
#        num_pnts = len(X_train)
#        w = Counter()
#        t = 0
#        epoch = 0
#        while epoch<self.max_epoch:
#            epoch += 1
#            for j in range(num_pnts):
#                print ('t={}'.format(t),end='\r')
#                t+=1
#                eta_t = 1/(t*self.reg_l)
#                if y[j]*dotProduct(w,X[j])<1:
#                    for f,v in w.items():
#                        w[f] = v*(1-eta_t*self.reg_l) 
#                    increment(w,eta_t*y[j],X[j])
#                else:
#                    for f,v in w.items():
#                        w[f] = v*(1-eta_t*self.reg_l) 
#        self.w_ = w
#        return self

##> Load data
X_train = load_data('X_train.pickle')
y_train = load_data('y_train.pickle')
X_val = load_data('X_val.pickle')
y_val = load_data('y_val.pickle')

##> Q6.8 
#reg_l_list = [1e-7,1e-5,1e-3,1e-1,1,1e1,1e2]
#reg_l_list = [0.1,0.3,1,3,10]
#score_list = []
#for reg_l in reg_l_list:
#    svm = pegasos(reg_l = reg_l,max_epoch=5)
#    svm.fit(X_train,y_train)
#    score = svm.score(X_val,y_val)
#    score_list.append(score)
#plt.semilogx(reg_l_list,score_list)
#plt.show()

##> Q6.9
#reg_l = 0.3
#svm = pegasos(reg_l = reg_l,max_epoch=5)
#svm.fit(X_train,y_train)
#predicts,correct = svm.predict(X_val,y_val)
#dig = np.digitize(np.abs(predicts),bins=np.arange(0,6))
#error_in_bin = []
#for i in np.arange(1,7):
#    selection = dig==i
#    error_in_bin.append(np.mean(np.array(correct)[selection]))
#plt.plot(range(6),error_in_bin)
#plt.show()


##> Q6.10
# Very rare to find ywX ~ 1. shoten the step size is a good idea

##> Q7.1
# reg_l is the selected lambda
# the second validation dataset is not correct, used for illustration
# after print out the first several words that has high abs(wi*xi),\
#        found words like 'the', 'of' etc often appear. Thus easy to lead to wrong prediction.
reg_l = 0.3
svm = pegasos(reg_l = reg_l,max_epoch=5)
svm.fit(X_train,y_train)
w = svm.w_
x = X_val[1]
forsave= []
for f,v in x.items():
    forsave.append([f,v,w[f],w[f]*v,abs(w[f]*v)])
df = pd.DataFrame(data=forsave,columns=['word','xi','wi','wi*vi','abs(wi*vi)']) 
df = df.sort_values(by=['abs(wi*vi)'],ascending=False)
print(df.head(50))

