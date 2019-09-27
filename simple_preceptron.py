#!/usr/bin/env python
# coding: utf-8

'''
Name : Omkar Thawakar, Alok jadhav
Perceptron Implementation for binary classification
'''

import numpy as np
from sklearn.utils import shuffle
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import json


np.random.seed(999999) #Setting a random seed is important for reproducibility

def read_libsvm(fname, num_features=0):
	
	data = []
	y = []
	row_ind = []
	col_ind = []
	with open(fname) as f:
		lines = f.readlines()
		for i, line in enumerate(lines):
			elements = line.split()
			y.append(int(elements[0]))
			for el in elements[1:]:
				row_ind.append(i)
				c, v = el.split(":")
				col_ind.append(int(c))
				data.append(float(v))
	if num_features == 0:
		num_features = max(col_ind) + 1
	X = csr_matrix((data, (row_ind, col_ind)), shape=(len(y), num_features))

	return X.toarray(), np.array(y), num_features


'''
    X: input vector or set of vectors
    w: weight vector
    b: bias
    
    Output: a numpy array containing the predictions produced by the linear threshold unit defined by w and b.
'''
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

'''
    X: set of input vectors
    y: set of labels
    w: weight vector
    b: bias
    
    Output: The accuracy achieved by the classifier defined by w and b on samples X.
'''
def accuracy(X, y, w, b):
    result = predict(X,w,b)
    unique, counts = np.unique(result == y, return_counts=True)
    res = dict(zip(unique, counts))
    return res[True]/len(y)


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
        prediction = predict(x,w,b)
        w = w + lr*y*x
        b = b + lr*y
    else:
        prediction = predict(x,w,b)
        w = w + lr*y*x
        
    return w,0.0


'''
    A class that will handle the storage of the historic values of training
'''
class History:
    def __init__(self, num_epochs):
        self.training_hist = dict()
        for n in range(num_epochs):
            self.training_hist[n] = {'w_hist': [], 
                                'b_hist': [], 
                                'acc_hist': [],
                                'point_hist':[]}
    def store(self, x, y, w, b, accuracy, epoch):
        self.training_hist[epoch]['point_hist'].append((x, y))
        self.training_hist[epoch]['w_hist'].append(w.copy())
        self.training_hist[epoch]['b_hist'].append(b)
        self.training_hist[epoch]['acc_hist'].append(accuracy)

def shuffle_arrays(X, y):
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]

def train(X_train, y_train, epochs=10, lr=0.01, use_bias=True):
    hist = History(epochs)
    w = np.random.uniform(0, 0.7, size=X_train.shape[1])      ##### initialize w
    b = 0.23                                                  ##### initialize bias
    update_flag = 0
    for epoch in range(epochs):
        
        X_train, y_train = shuffle(X_train,y_train)   ##### shuffle training samples
        
        for x,y in zip(X_train,y_train):
            
            predicted = predict(x,w,b)                      #### predict label
            
            if predicted*y < 0 :                            #### if mistake
                
                w,b = update(x,y,w,b,lr, b_update=use_bias)
                update_flag+=1
                
        acc = accuracy(X_train, y_train, w, b) 
        hist.store(x, y, w, b, acc, epoch)           #### store history
                
    return w, b, hist, update_flag


# ### Importing data from libsvm format
X_train, y_train, num_features = read_libsvm('data/data_train')


# ### Train the classifier:

# (a) First, note that in the formulation above, the bias term b is explicitly mentioned.
# This is because the features in the data do not include a bias feature. Of course,
# you could choose to add an additional constant feature to each example and
# not have the explicit extra b during learning. (See the class lectures for more
# information.) However, here, we will see the version of Perceptron that explicitly
# has the bias term.

# In[94]:


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
        w, b, hist,updates = train(x_train, y_train, epochs=10, lr=0.01, use_bias=True)
        acc = accuracy(X_test, y_test, w, b)
        print('For lr {} accuracy of perceptron is {} and No of updates are {} .'.format(lr, acc, updates))
        
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

# #### (a). Best Hyper parameters


print('Learning rate ::: ',results[1]['lr'][tmp])
print('='*50)
print('Learned Weights are :: ',results[1]['perceptron'][results[1]['lr'][tmp]]['w'])
print('='*50)
print('Learned Bias is :: ',results[1]['perceptron'][results[1]['lr'][tmp]]['b'])
print('='*50)


# #### (b) The cross-validation accuracy for the best hyperparameter

print('For lr {} Maximum Average Accuracy {} '.format(results[1]['lr'][tmp], avg_acc))




X_train, y_train, num_features = read_libsvm('data/data_train')
np.random.seed(232415)
epochs=20
w, b, hist, updates = train(X_train, y_train, epochs=epochs, lr=results[1]['lr'][tmp], use_bias=True)
print('Accuracy of Perceptron on Train Set is {}.'.format(hist.training_hist[epochs-1]['acc_hist']))


# #### (e) Test set accuracy

X_test, y_test, num_features = read_libsvm('data/data_test')
print('For lr {} accuracy of perceptron is {} .'.format(lr, accuracy(X_test, y_test, w, b)))


# #### (f) Plot a learning curve where the x axis is the epoch id and the y axis is the training set accuracy using the classier (or the averaged classier, as appropriate) at the end of that epoch. Note that you should have selected the number of epochs using the learning curve (but no more than 20 epochs).

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


# # 4.3 from report

X_train, y_train, num_features = read_libsvm('data/data_train')
np.random.seed(232415)
epochs=20
w, b, hist, updates = train(X_train, y_train, epochs=epochs, lr=results[1]['lr'][tmp], use_bias=True)
print('Accuracy of Perceptron on Train Set is {}.'.format(hist.training_hist[epochs-1]['acc_hist']))

max_w = sorted(w)[-10:]
min_w = sorted(w)[:10]

max_idx = []
for element in max_w:
    max_idx.append(np.where(w == element))

min_idx = []
for element in min_w:
    min_idx.append(np.where(w == element))


filename = 'data/vocab_idx.json'

if filename:
    with open(filename, 'r') as f:
        datastore = json.load(f)


print('10 words with the highest weights are')
for idx in max_idx:
    print(datastore[str(idx[-1][-1])])
print('='*50)

print('10 words with the lowest weights are')
for idx in min_idx:
    print(datastore[str(idx[-1][-1])])
print('='*50)



