"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is linear_regression_noreg, linear_regression_invertible，regularized_linear_regression,
tune_lambda, test_error and mapping_data.
"""

import numpy as np
import pandas as pd
import numpy.linalg as lg

def mean_absolute_error(w, X, y):
    """
    Compute the mean absolute error on test set given X, y, and model parameter w.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test feature.
    - y: A numpy array of shape (num_samples, ) containing test label
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean absolute error
    """
    err = None
    pred = X.dot(w)
    err = np.mean(np.absolute(pred - y))
    return err

def linear_regression_noreg(X, y):
  """
  Compute the weight parameter given X and y.
  Inputs:
  - X: A numpy array of shape (num_samples, D) containing feature.
  - y: A numpy array of shape (num_samples, ) containing label
  Returns:
  - w: a numpy array of shape (D, )
  """
  w = None

  w = lg.inv(X.T.dot(X)).dot(X.T).dot(y)
  return w

def linear_regression_invertible(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    w = None
    mat = X.T.dot(X)
    Identity = 0.1 * np.identity(mat.shape[0])
    eigen_val, eigen_vec = lg.eig(mat)

    while(eigen_val).min() < 0.00001:
        mat += Identity
        eigen_val, eigen_vec = lg.eig(mat)

    w = lg.inv(mat).dot(X.T).dot(y)
    return w


def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    - lambd: a float number containing regularization strength
    Returns:
    - w: a numpy array of shape (D, )
    """
    w = None

    mat = X.T.dot(X)
    Identity = lambd * np.identity(mat.shape[0])

    mat += Identity

    w = lg.inv(mat).dot(X.T).dot(y)
    return w

def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training feature.
    - ytrain: A numpy array of shape (num_training_samples, ) containing training label
    - Xval: A numpy array of shape (num_val_samples, D) containing validation feature.
    - yval: A numpy array of shape (num_val_samples, ) containing validation label
    Returns:
    - bestlambda: the best lambda you find in lambds
    """
    bestlambda = None
    min_mse = 1000000

    for i in range(-19, 20):
        lambd = 10**i
        w = regularized_linear_regression(Xtrain, ytrain, lambd)
        e = mean_absolute_error(w, Xval, yval)

        if(e < min_mse):
            bestlambda = lambd
            min_mse = e

    return bestlambda


def mapping_data(X, power):
    """
    Mapping the data.
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training feature.
    - power: A integer that indicate the power in polynomial regression
    Returns:
    - X: mapped_X, You can manully calculate the size of X based on the power and original size of X
    """
    map_mat = [X]

    for i in range(2, power+1):
        map_mat.append(X**power)
    X = np.concatenate(map_mat, axis=1)
    return X


