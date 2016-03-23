"""This module generates data for logistic regression with linear sepparating surface"""
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import scipy.optimize as op
from matplotlib.legend_handler import HandlerLine2D
from pylab import *
import time

#parameters
gamma = 1e-4
dim = 10
data_size = 1e6
mu, sigma1, sigma2 = 0, 10, 5
x_interval = 10
random_seed_w0 = 31
random_seed_w = 3
random_seed_x = 91
data_name, m, n = 'logistic regression', dim, data_size

def plotdata(x, y):
    if x.shape[1] == 3:
        loc_y = y.reshape((y.shape[0],))
        plt.plot(x[loc_y == 1, 0], x[loc_y == 1, 1],'ro')
        plt.plot(x[loc_y == -1, 0], x[loc_y == -1, 1],'bx')

def drawline(w, x_interval, color):
    if w.size == 3:
        x0_left = -x_interval / 2;
        x0_right = x_interval / 2;
        x1_left = -(w[0] * x0_left + w[2]) / w[1]
        x1_right = -(w[0] * x0_right + w[2]) / w[1]
        plt.plot([x0_left, x0_right], [x1_left, x1_right], color)
        plt.axis((-x_interval / 2, x_interval / 2,-x_interval / 2,x_interval / 2))

def loss(w, x, y):
    res = (np.sum(np.log(np.exp(-y * np.dot(x, w)) + np.ones(y.shape)))) / y.shape[0] \
    + gamma * np.square(np.linalg.norm(w))
    return res

def grad(w, x, y):
    s = np.exp(y * np.dot(x, w))
    s = 1 / (1 + s)
    res = (-(s * x * y))
    res = ((np.sum(res, axis = 0)).reshape((x.shape[1], 1)) / y.shape[0] + 2*gamma*w)
    return res

def batch_loss(w, x, y, i1, i2):
    x_loc = x[i1 : i2]
    y_loc = y[i1 : i2]
    res = (np.sum(np.log(np.exp(-y_loc * np.dot(x_loc, w)) + np.ones(y_loc.shape)))) / y_loc.shape[0] \
    + gamma * np.square(np.linalg.norm(w))
    return res

def batch_grad(w, x, y, i1, i2):
    x_loc = x[i1 : i2]
    y_loc = y[i1 : i2]
    s = np.exp(y_loc * np.dot(x_loc, w))
    s = 1 / (1 + s)
    res = (-(s * x_loc * y_loc))
    res = ((np.sum(res, axis = 0)).reshape((x_loc.shape[1], 1)) / y_loc.shape[0] + 2*gamma*w)
    return res

def one_func_grad(w, x, y, i):
    y1, x1 = y[i], x[i]
    s = 1 / (1 + np.exp(y1 * np.dot(x1, w)))
    res = ((-(s * x1 * y1)).reshape((x.shape[1], 1)) + 2*gamma*w)
    return res

def one_func_loss(w, x, y, i):
    res = (np.log(np.exp(-y[i] * np.dot(x[i], w)) + 1)) + gamma * np.square(np.linalg.norm(w))
    return res

def test_err(w, x, y):
    my_y = np.sign(np.dot(x, w))
    res = np.abs(y - my_y)
    res = (np.sum(res)/2)
    res /= y.size
    res = 1 - res
    return res

seed(random_seed_w)
w = np.random.normal(mu, sigma1, (dim + 1, 1))

seed(random_seed_x)
x = np.concatenate((np.random.rand(data_size, dim) * x_interval - x_interval / 2,
                np.ones((data_size, 1))), axis = 1)
y = np.sign(np.dot(x, w))

#  Generating the starting point
seed(random_seed_w0)
w0 = np.random.normal(mu, sigma1, (dim + 1, 1))

opt_res = op.minimize(fun=lambda w: loss(w.reshape((w.size, 1)), x, y), x0=w0, tol=1e1)
true_loss = opt_res['fun']
if __name__ == "__main__":
    print("Generating w: ", w)
    print("Starting point: ", w0)
    print("\nTrue loss: ", true_loss)

# drawline(opt_res['x'], x_interval, '-r')
# plotdata(x, y)
# plt.show()