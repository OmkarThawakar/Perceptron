#!/usr/bin/env python

"""
Name : Omkar Thawakar

"""

import numpy as np
from scipy.sparse import csr_matrix

def generate_data(num_samples):
    size = num_samples // 2
    x1 = np.random.multivariate_normal([0, 0], np.eye(2), size)
    y1 = -np.ones(size).astype(int)
    x2 = np.random.multivariate_normal([3, 3], np.eye(2), size)
    y2 = np.ones(size).astype(int)
    
    X = np.vstack((x1, x2))
    y = np.append(y1, y2)
    
    return X, y

def plot(x, y):
    fig = plt.figure(figsize = (7, 5), dpi = 100, facecolor = 'w')
    plt.scatter(x[:, 0], x[:, 1], c=y, edgecolor='black', cmap=cmap)
    plt.show()


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
    result = predict(X,w,b)
    unique, counts = np.unique(result == y, return_counts=True)
    res = dict(zip(unique, counts))
    return res[True]/len(y)

def update(x, y, w, b, lr):
    prediction = predict(x,w,b)
    w = w + lr*y*x
    b = b + lr*y
    return w,b
    
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

def store(self, x, y, w, b, accuracy, epoch):
        self.training_hist[epoch]['point_hist'].append((x, y))
        self.training_hist[epoch]['w_hist'].append(w.copy())
        self.training_hist[epoch]['b_hist'].append(b)
        self.training_hist[epoch]['acc_hist'].append(accuracy)


def train(X_train, y_train, epochs=10, lr=0.01):
    hist = History(epochs)
    w = np.random.uniform(0, 1, size=X_train.shape[1])      ##### initialize w
    b = 0                                                   ##### initialize bias
    
    for epoch in range(epochs):
    
        X_train, y_train =shuffle_arrays(X_train,y_train)   ##### shuffle training samples
        
        for x,y in zip(X_train,y_train):
            
            predicted = predict(x,w,b)                      #### predict label
            
            if predicted == y :                             #### if mistake
                
                w,b = update(x,y,w,b,lr)
                
        acc = accuracy(X_train, y_train, w, b) 
        hist.store(x, y, w, b, acc, epoch)           #### store history
                
    return w, b, hist

def read_libsvm(fname, num_features=0):
	from scipy.sparse import csr_matrix
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




