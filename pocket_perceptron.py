#!/usr/bin/env python
'''
Name : Omkar Thawakar, Alok Jadhav
Pocket Perceptron Implementation for binary classification
'''

import numpy as np
from sklearn.utils import shuffle
from scipy.sparse import csr_matrix
from Perceptron import *


'''
    x: input vector
    y: label
    w: weight vector
    b: bias
    lr: learning rate
    
    Updates the w and b according to the Perceptron update rule.
    
    Output: updated w and b
'''
def update(x, y, w, b, lr, b_update=True):
    if b_update:
        w = w + lr*y*x
        b = b + lr*y
    else:
        w = w + lr*y*x
        b=0
        
    return w,b
    

def predict(X, w, b):
    result = np.array([])
    if len(X.shape) > 1 :
        for x in X:
            if np.dot(x,w) + b >= 0 :
                result = np.append(result,1)
            else:
                result = np.append(result,-1)
    else:
        if np.dot(X,w) + b >= 0 :
            return 1.
        else:
            return -1.
    return result


def accuracy(X, y, w, b):
    result = 0
    for sample,label in zip(X,y):
        correct,incorrect = 0,0
        for w_,b_ in zip(w,b):
            predicted = predict(sample,w_,b_)
            if predicted == label:
                correct+=1
            else:
                incorrect+=1
        if correct>incorrect :
            result+=1
    return result/len(X)
        

def train(X_train, y_train, epochs=10, lr=0.01, use_bias=True):
    hist = History(epochs)
    w = np.array([np.random.uniform(0, 0.78, size=X_train.shape[1])] )     ##### initialize w
    b = np.array([0] )                        ##### initialize bias
    
    w = w.tolist()
    b = b.tolist()
    
    update_flag = 0
    for epoch in range(epochs):
        
        X_train, y_train = shuffle(X_train,y_train)   ##### shuffle training samples
        
        t = 1
        step = 0
        for x,y in zip(X_train,y_train):
            
            predicted = predict(x,w[-1],b[-1])               #### predict label
            
            if predicted*y < 0 :                             #### if mistake
                w_,b_ = update(x,y,w[-1],b[-1],lr, b_update=use_bias)
                w.append(w_)
                b.append(b_)
                update_flag+=1
                          
        acc = accuracy(X_train, y_train, w, b) 
        hist.store(x, y, w, b, acc, epoch)           #### store history
                
    return w, b, hist, update_flag


# ### Train the classifier:

results = {1:{'lr':[],'acc':[],'perceptron':{1:{},0.1:{},0.01:{}}},
           2:{'lr':[],'acc':[],'perceptron':{1:{},0.1:{},0.01:{}}},
           3:{'lr':[],'acc':[],'perceptron':{1:{},0.1:{},0.01:{}}},
           4:{'lr':[],'acc':[],'perceptron':{1:{},0.1:{},0.01:{}}},
           5:{'lr':[],'acc':[],'perceptron':{1:{},0.1:{},0.01:{}}},
          }

flag=1
folds = ['fold1','fold2','fold3','fold4','fold5']
for fold in folds:
    print('Cross Validation fold :: ',fold)
    cross_folds = ['fold1','fold2','fold3','fold4','fold5']
    cross_folds.remove(fold)
    print('Training folds :: ',cross_folds)
    x_train,y_train = [],[]
    for file in cross_folds:
        if len(x_train) != 0 and len(y_train) != 0:
            tmp1_train, tmp2_train, num_features = read_libsvm('data/CVfolds/{}'.format(file))
            
            x_train = np.concatenate((x_train,tmp1_train),axis=0)
            y_train = np.concatenate((y_train,tmp2_train),axis=0)
        else:
            x_train, y_train, num_features = read_libsvm('data/CVfolds/{}'.format(file))            

    X_test, y_test, num_features = read_libsvm('data/CVfolds/{}'.format(fold))
    for lr in [1,0.1,0.01]:
        w, b, hist, updates = train(x_train, y_train, epochs=10, lr=lr, use_bias=True)
        acc = accuracy(X_test, y_test, w, b)
        print('For lr {} accuracy of perceptron is {} with total updates {} .'.format(lr, acc, updates))
        
        results[flag]['lr'].append(lr)
        results[flag]['acc'].append(acc)
        results[flag]['perceptron'][lr]['w'] = w
        results[flag]['perceptron'][lr]['b'] = b
        results[flag]['perceptron'][lr]['hist'] = hist
    flag+=1
        
    print('='*50)

avg_acc = -1
tmp = -1
for lr_id in [i for i in range(len(results[1]['lr']))]:
    for fold in [1,2,3,4,5]:
        acc = 0
        for key in list(results.keys()):
            acc = acc + results[key]['acc'][lr_id]
        if avg_acc < acc/len(list(results.keys())) :
            avg_acc = acc/5
            tmp = lr_id
        else:
            pass
print('For lr {} Maximum Average Accuracy {} '.format(results[1]['lr'][tmp], avg_acc))


lr = results[1]['lr'][tmp]   ##### most likely learning rate from cross validation


print('Learning rate ::: ',lr)
print('='*50)
print('Learned Weights are :: ',results[1]['perceptron'][results[1]['lr'][tmp]]['w'])
print('='*50)
print('Learned Bias is :: ',results[1]['perceptron'][results[1]['lr'][tmp]]['b'])
print('='*50)


print('For lr {} Maximum Average Accuracy {} '.format(lr, avg_acc))


X_train, y_train, num_features = read_libsvm('data/data_train')
np.random.seed(232415)
epochs=20
w, b, hist, updates = train(X_train, y_train, epochs=epochs, lr=lr, use_bias=True)
print('Accuracy of Perceptron on Train Set is {}.'.format(hist.training_hist[epochs-1]['acc_hist']))


X_test, y_test, num_features = read_libsvm('data/data_test')
print('For lr {} accuracy of perceptron is {} .'.format(lr, accuracy(X_test, y_test, w, b)))


import matplotlib.pyplot as plt

acc = []
epochs = []
for key in list(hist.training_hist.keys()):
    epochs.append(key)
    acc.append(hist.training_hist[key]['acc_hist'][-1])
    
plt.plot(epochs,acc)
plt.title('Plot of epochs vs Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()    


X_train, y_train, num_features = read_libsvm('data/data_train')
np.random.seed(232415)
epochs=20
w, b, hist, updates = train(X_train, y_train, epochs=epochs, lr=lr, use_bias=True)

for epoch in range(epochs):
    print('Accuracy of Perceptron for epoch {} is {}.'.format(epoch+1,hist.training_hist[epoch]['acc_hist']))
print('='*50)


X_test, y_test, num_features = read_libsvm('data/data_test')
print('For lr {} accuracy of perceptron is {} .'.format(lr, accuracy(X_test, y_test, w, b)))
