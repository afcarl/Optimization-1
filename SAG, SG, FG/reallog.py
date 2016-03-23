"""This module is for importing datasets for logistic regression"""
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import scipy.optimize as op
from matplotlib.legend_handler import HandlerLine2D
import time
from logreg import true_loss, grad, loss, batch_loss, batch_grad, test_err
from sklearn.datasets import load_svmlight_file
from pylab import *

#  Parameters
mu, sigma1 = 0, 10
random_seed_w0 = 32

start = time.clock()

# x_d, y_d = load_svmlight_file('datasets/fourclass_scale.txt')
# data_name = 'Fourclass'
# x_d, y_d = load_svmlight_file('datasets/australian_scale.txt')
# data_name = 'Australian'
# x_d, y_d = load_svmlight_file('datasets/colon-cancer.txt')
# data_name = 'Colon-cancer'
# x_d, y_d = load_svmlight_file('datasets/ijcnn1.txt')
# data_name = 'IJCNN'
# true_loss = 0.23256250109316198
# x_d, y_d = load_svmlight_file('datasets/mnist_scale.txt')
# data_name = 'Mnist'
# true_loss = 0
x_d, y_d = load_svmlight_file('datasets/SUSY.txt')
data_name = 'SUSY'
true_loss = 0 # 2.3828995645457518

x = np.concatenate((x_d.toarray(), np.ones((x_d.shape[0], 1))), axis=1)
y = y_d.reshape((y_d.size, 1))
dim = x.shape[1]
y[y != 1] = -1
print(time.clock() - start)
seed(random_seed_w0)
w0 = np.random.normal(mu, sigma1, (dim, 1))

# data_size = y.size
n = y.size
m = x.shape[1]

# opt_res = op.minimize(fun=lambda w: loss(w.reshape((w.size, 1)), x, y), x0=w0, tol=1e-10, options={'maxiter': 100})
# print(time.clock() - start)
# true_loss = opt_res['fun']
# print("\nTrue loss: ", true_loss)

if __name__ == "__main__":
    print("\nTrue loss: ", true_loss)
