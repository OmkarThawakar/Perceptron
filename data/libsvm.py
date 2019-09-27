from scipy.sparse import csr_matrix
import numpy as np

def read_libsvm(fname, num_features=0):
	'''
		Reads a libsvm formatted data and outputs the training set (sparse matrix)[1], 
		the label set and the number of features. The number of features
		can either be provided as a parameter or inferred from the data.

		Example usage:
		
		X_train, y_train, num_features = read_libsvm('data_train')
		X_test, y_test, _ = read_libsvm('data_test', num_features)

		[1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
	'''
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

	return X, np.array(y), num_features
