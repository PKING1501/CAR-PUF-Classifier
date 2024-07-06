import numpy as np
import sklearn
from scipy.linalg import khatri_rao
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
import time as tm

# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def my_fit( X_train, y_train ):
################################
#  Non Editable Region Ending  #
################################
	feat = my_map(X_train)
	model = LogisticRegression(penalty='l2', C=1.0, fit_intercept=False, solver='lbfgs', max_iter=2500)
	# Use this method to train your model using training CRPs
	# X_train has 32 columns containing the challeenge bits
	# y_train contains the responses
	model.fit(feat, y_train)
	w = model.coef_[0]
	b = 0
	
	# THE RETURNED MODEL SHOULD BE A SINGLE VECTOR AND A BIAS TERM
	# If you do not wish to use a bias term, set it to 0
	return w, b


################################
# Non Editable Region Starting #
################################
def my_map( X ):
################################
#  Non Editable Region Ending  #
################################
	X = np.hstack((X, np.ones((X.shape[0], 1))))
	feat = khatri_rao(X.T, X.T).T
	# Use this method to create features.
	# It is likely that my_fit will internally call my_map to create features for train points
	
	return feat

Z_trn = np.loadtxt( "train.dat" )
Z_tst = np.loadtxt( "test.dat" )

n_trials = 5

d_size = 0
t_train = 0
t_map = 0
acc = 0

for t in range( n_trials ):
	tic = tm.perf_counter()
	w, b = my_fit( Z_trn[:, :-1], Z_trn[:,-1] )
	toc = tm.perf_counter()
	t_train += toc - tic

	d_size += w.shape[0]

	tic = tm.perf_counter()
	feat = my_map( Z_tst[:, :-1] )
	toc = tm.perf_counter()
	t_map += toc - tic

	scores = feat.dot( w ) + b
	print( scores)
	pred = np.zeros_like( scores )
	pred[scores > 0] = 1
	acc += np.average( Z_tst[ :, -1 ] == pred )

d_size /= n_trials
t_train /= n_trials
t_map /= n_trials
acc /= n_trials

print( d_size, t_train, t_map, acc )
