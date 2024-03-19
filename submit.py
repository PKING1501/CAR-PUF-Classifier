import numpy as np
import sklearn
from scipy.linalg import khatri_rao
from sklearn.linear_model import LogisticRegression

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
	model = LogisticRegression(penalty='l2', C=120, fit_intercept=True, max_iter=2500)
	# Use this method to train your model using training CRPs
	# X_train has 32 columns containing the challeenge bits
	# y_train contains the responses
	model.fit(feat, y_train)
	w = model.coef_[0]
	b = model.intercept_
	
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
	X = 1 - 2 * X
	
	arr = []
	for i in range(X.shape[0]):
		temp = []
		temp.append(X[i][-1])
		for j in range(X.shape[1] - 2, -1, -1):
			temp.append(X[i][j] * temp[-1])
		temp.reverse()
		arr.append(np.array(temp))
    
	arr = np.array(arr)
	
	feat = []
	for i in range(X.shape[0]):
		res = []
		for j in range(X.shape[1]):
			for k in range(j+1, X.shape[1]):
				res.append(arr[i][j] * arr[i][k])
		feat.append(np.array(res))

	feat = np.array(feat)
	feat = np.hstack((feat, np.ones((feat.shape[0], 1))))

	return feat