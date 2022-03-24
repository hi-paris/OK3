![Build Status](https://github.com/hi-paris/OK3/workflows/pytesting/badge.svg) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Github All Releases](https://img.shields.io/github/downloads/hi-paris/OK3/total.svg)]()
_#Waiting for "pytesting" upcoming validation_

# pythonOK3
classes and methods to implement the OK3 method : decision trees with a kernelized output for structured prediction.

To easily test OK3, it is built in order to take as argument output vectors matrixes rather than Gram matrixes only.
It will build internally the Gram output matrix and return it to the 'true" fit and predict methods.

To test the OK3 trees with multilabel classification or regression on vector data (with Gini impurity and variance).
Numerous tests are drafted in tests/test_tree_clf_and_reg.py and tests/tests_complementary.py

## Protocol :

1 - Clone 'pythonOK3' project

2 - In cloned repertory, execute in terminal : `python setup.py build_ext --inplace`
    It will compile the Cython files, et possibly create warnings (to ignore)

3 - Eventually restart the Python kernel to take account of the compiled files changes.

4 - To execute the tests, input in terminal : `pytest tests/test_tree_clf_and_reg.py`
    It will execute the file's tests (Total lenght sub 11 min)
    These tests are used on regression and classification tasks only, and tests on structured problems are still to develop.

5 - Another way to quickly test those classification and regression functions and to compare the results with the classic classification and regression trees is to execute the files "test_classification.py" and "test_regression.py" that will print lines demonstrating the uniformity of classic and OK3 trees.

6 - To test the structured prediction (different than simple classification and regression), cf. file "exemple_utilisation.py" (or check below) that describes how to use the OK3 trees (on a multilabel classification close to a structured prediction problem).

## Usecase of OK3 Trees

```python
############################################################
# # # # #           How to use OK3 Trees           # # # # #
############################################################

from _classes import OK3Regressor, ExtraOK3Regressor
from kernel import *
from sklearn import datasets


#%% Generate a dataset with outputs being long vectors of 0 and 1

n_samples=4000
n_features=100
n_classes=1000
n_labels=2

# Fit on a half, 
# Testing on a quarter, 
# Last quarter used as possible outputs' ensemble

# It is a big dataset for the algorithm, so it is kind of slow,
# Using parameters regulating the tree's growth will help:
# --> max_depth, max_features 

X, y = datasets.make_multilabel_classification(n_samples=n_samples, 
                                               n_features=n_features, 
                                               n_classes=n_classes, 
                                               n_labels=n_labels)

# First half is the training set
X_train = X[:n_samples//2]
y_train = y[:n_samples//2]

# Third quarter is the test set
X_test = X[n_samples//2 : 3*n_samples//4]
y_test = y[n_samples//2 : 3*n_samples//4]

# The last quarter of outputs is used to have candidates for the tree's decoding
# Predictions will be in this ensemble
y_candidates = y[3*n_samples//4:]


#%% Fitting one (two) tree(s) to the data

# A kernel must be chosen to be used on output datas in vectorial format above

# For now the list of usable kernels is :
kernel_names = ["linear", "mean_dirac", "gaussian", "laplacian"]

# Gaussian and exponential kernels have a parameter gamma to the the width of the kernel


# Let's choose a kernel
# We can indifferently input the name (and its potential parameters) :
kernel1 = ("gaussian", .1) # ou bien kernel1 = "linear"
# Or
kernel2 = Mean_Dirac_Kernel()

# Then we can create our estimator, that will work calculating gaussian kernels between outputs :
ok3 = OK3Regressor(max_depth=6, max_features='sqrt', kernel=kernel1) 

# It is also while creating the estimator that we can fulfill 
# the maximum depth, the impurity's minimal reduction on each split, 
# The minimal number of samples in each leaf, etc as in classical trees

# We can now fit the estimator to our training data :
ok3.fit(X_train, y_train)
# We could also input a "sample_weight" parameter : a weight vector, positive or null on the training examples: 
# A weight "0" in an example means that the example will not be taken into accountun
print("check")
# ALTERNATIVE : we can also fulfill only the computation mode of the kernels during the 'fit' 
# allowing us to change the kernel with the same estimator
# ex:
extraok3 = ExtraOK3Regressor(max_depth=6, max_features='sqrt')
extraok3.fit(X_train, y_train, kernel=kernel2) # the estimator will keep in memory this new kernel


#%% (OPTIONAL) Before decoding the tree

# We can :
# get the tree's depth
depth = ok3.get_depth()
# get the number of leaves
n_leaves = ok3.get_n_leaves()
# get the predictions of each leaf with weight vectors on training outputs
leaves_preds_as_weights = ok3.get_leaves_weights()
# get the leaves with new data
X_test_leaves = ok3.apply(X_test)
# get the predictions for new data with weights
test_weights = ok3.predict_weights(X_test)
# Compute the R2 score of our predictions in Hilbert's Space in which are set the outputs
r2_score = ok3.r2_score_in_Hilbert(X_test, y_test) # We can specify a vector sample_weight


#%% Deconding the tree(s) for outputs' prediction

# For decoding, we can propose an ensemble of candidates' outputs, 
# or we can "None" it. In this case the candidates' outputs are the training example's outputs
candidates_sets = [None, y_candidates]
# We can try with candidates=y_candidates or with nothing (ou None)

# We can either decode the tree considering an ensemble of candidates' outputs
leaves_preds = ok3.decode_tree(candidates=y_candidates)
# Then be able to predict quickly the outputs
y_pred_1 = ok3.predict(X_test[::2])
y_pred_2 = ok3.predict(X_test[1::2])

print("check")

# Or we can decode for a serie of inputs, by fulfilling the candidates
y_pred_extra_1 = extraok3.predict(X_test[::2], candidates=y_candidates)
y_pred_extra_2 = extraok3.predict(X_test[1::2]) # Remember the predictions from each leaf


#%% Performance's evaluation

# We can calculate the R2 score in the Hilbert's Space as said before (no need for decoding):
r2_score = extraok3.r2_score_in_Hilbert(X_test, y_test) # We can specify a 'sample_weight' vector

# We can compute scores on the real decoded datas :
hamming_score = ok3.score(X_test, y_test, metric="hamming") # We can specify a 'sample_weight' vector
# Note : It is not necessary to fulfill again a candidates' ensemble since OK3 already got one

# One of the KPI available for this score function is the 'top k accuracy'.
# To get it we need to fulfill the candidates' ensemble in the function on each call:
top_3_score = ok3.score(X_test, y_test, candidates=y_candidates, metric="top_3") 
# We can fulfill any int rather than the "3" above :
top_11_score = ok3.score(X_test, y_test, candidates=y_candidates, metric="top_11") 


#%% Miscellaneous things to know

##### About 'candidates' #####

# When the candidates ensembles are required, they are not provided, 
# So the research is done with the training dataset's outputs' ensemble,  
# memorized by the tree (faster).

# The candidates can be informed in the functions 'decode_tree', 'predict', 
# 'decode' (which is equal to 'predict), and in 'score'.
# Once that an ensemble of candidates is fulfilled in one of this functions, the ensemble of 
# possible predictions on leafs is memorized by the estimator (but not the ensemble of candidates 
# itself because too big) , that's why
# it is not mandatory and even unwishable compute-wise to inform several times 
# the same ensemble of candidates. The first time is enough. 
# On the other hand when we compute a top k accuracy score then it is mandatory 
# to inform an ensemble of candidates because the decoding is different 
# if we need to return several predictions for each leaf.

```

## Usecase of OK3 forests

```python
###########################################################
# # # # #   How to use OK3's forests   # # # # #
############################################################

from _forest import RandomOKForestRegressor, ExtraOKTreesRegressor
from kernel import *
from sklearn import datasets


#%% Generate a dataset with outputs being big vectors of 0 and 1

n_samples=1000
n_features=100
n_classes=1000
n_labels=2

# Fit on a half, 
# Testing on a quarter, 
# Last quarter used as possible outputs' ensemble

# It is a big dataset for the algorithm, so it is kind of slow,
# Using parameters regulating the tree's growth will help:
# --> max_depth, max_features s 

X, y = datasets.make_multilabel_classification(n_samples=n_samples, 
                                               n_features=n_features, 
                                               n_classes=n_classes, 
                                               n_labels=n_labels)

# First half is the training set
X_train = X[:n_samples//2]
y_train = y[:n_samples//2]

# Third quarter is the test set
X_test = X[n_samples//2 : 3*n_samples//4]
y_test = y[n_samples//2 : 3*n_samples//4]

# The last quarter of outputs is used to have candidates for the tree's decoding
# Predictions will be in this ensemble
y_candidates = y[3*n_samples//4:]


#%% Fit a forest to the data

# A kernel must be chosen to be used on output datas in vectorial format above

# For now the list of usable kernels is :
kernel_names = ["linear", "mean_dirac", "gaussian", "laplacian"]

# Gaussian and exponential kernels have a parameter gamma to the the width of the kernel

# Let's choose a kernel
# We can indifferently input the name (and its potential parameters) :
kernel1 = ("gaussian", .1) # ou bien kernel1 = "linear"

# Then we can create our estimator, that will work calculating gaussian kernels between outputs :
okforest = RandomOKForestRegressor(n_estimators=20, max_depth=6, max_features='sqrt', kernel=kernel1) 

# It is also while creating the estimator that we can fulfill 
# the maximum depth, the impurity's minimal reduction on each split, 
# The minimal number of samples in each leaf, etc as in classical trees

# We can now fit the estimator to our training data :
okforest.fit(X_train, y_train)
# We could also input a "sample_weight" parameter : a weight vector, positive or null on the training examples: 
# A weight "0" in an example means that the example will not be taken into account

#%% (OPTIONAL) Before decoding the tree

# We can get the R2 score of our predictions in the Hilbert space in which are set the outputs
r2_score = okforest.r2_score_in_Hilbert(X_test, y_test) # We can specify a vector sample_weight


#%% Outputs' prediction

# For decoding, we can propose an ensemble of candidates' outputs, 
# or we can "None" it. In this case the candidates' outputs are the training example's outputs
candidates_sets = [None, y_candidates]
# We can try with candidates=y_candidates or with nothing (ou None)

# Or we can decode for a serie of inputs, by fulfilling the candidates
y_pred_1 = okforest.predict(X_test[::2], candidates=y_candidates)
y_pred_2 = okforest.predict(X_test[1::2])  # Remember the predictions from each leaf


#%% Performance's evaluation

# We can calculate the R2 score in the Hilbert's Space as said before (no need for decoding):
r2_score = okforest.r2_score_in_Hilbert(X_test, y_test) # on peut y spécifier un vecteur sample_weight

# We can compute scores on the real decoded datas :
hamming_score = okforest.score(X_test, y_test, metric="hamming") # on peut y spécifier un vecteur sample_weight
# Note : It is not necessary to fulfill again a candidates' ensemble since OK3 already got one

# One of the KPI available for this score function is the 'top k accuracy'.
# To get it we need to fulfill the candidates' ensemble in the function on each call:
top_3_score = okforest.score(X_test, y_test, candidates=y_candidates, metric="top_3") 
# We can fulfill any int rather than the "3" above :
top_11_score = okforest.score(X_test, y_test, candidates=y_candidates, metric="top_11") 


#%% Miscellaneous things to know

##### About 'candidates' #####

# When the candidates ensembles are required, they are not provided, 
# So the research is done with the training dataset's outputs' ensemble,  
# memorized by the tree (faster).

# The candidates can be informed in the functions 'decode_tree', 'predict', 
# 'decode' (which is equal to 'predict), and in 'score'.
# Once that an ensemble of candidates is fulfilled in one of this functions, the ensemble of 
# possible predictions on leafs is memorized by the estimator (but not the ensemble of candidates 
# itself because too big) , that's why
# it is not mandatory and even unwishable compute-wise to inform several times 
# the same ensemble of candidates. The first time is enough. 
# On the other hand when we compute a top k accuracy score then it is mandatory 
# to inform an ensemble of candidates because the decoding is different 
# if we need to return several predictions for each leaf.
```
