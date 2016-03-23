"""This module is for importing datasets for linear regression"""
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import scipy.optimize as op
from matplotlib.legend_handler import HandlerLine2D
import time
from linreg import true_loss, grad, loss, batch_loss, batch_grad
from sklearn.datasets import load_svmlight_file
from pylab import *

#Parameters
random_seed_w0 = 32
mu, sigma1 = 0, 10

x_d, y_d = load_svmlight_file('datasets/abalone_scale.txt')
x = np.concatenate((x_d.toarray(), np.ones((x_d.shape[0], 1))), axis=1)
y = y_d.reshape((y_d.size, 1))
dim = x.shape[1]
data_name = 'Abalone'
#Generating the starting point
seed (random_seed_w0)
w0 = np.random.normal(mu, sigma1, (dim, 1))
n = y.size
m = x.shape[1]

opt_res = op.minimize(fun=lambda w: loss(w.reshape((w.size, 1)), x, y), x0=w0, tol=1e-6) #, options={'maxiter': 30})
true_loss = opt_res['fun']
if __name__ == "__main__":
    print("\nTrue loss: ", true_loss)