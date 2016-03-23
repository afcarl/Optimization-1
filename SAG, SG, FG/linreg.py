"""This module generates data for linear regression and provides some functions for optimization"""
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import scipy.optimize as op
from matplotlib.legend_handler import HandlerLine2D
from pylab import *
import time

#Parameters
dim = 100
data_size = 1e3
random_seed_w0 = 32
random_seed_w = 223
random_seed_x = 44
random_seed_noise = 113
mu, sigma1, sigma2 = 0, 10, 8
x_interval = 10
data_name, m, n = 'linear regression', dim, data_size

#Functions
def plotdata(x, y):
    if x.shape[1] == 2:
        plt.plot(x[:, 0], y,'ro')

def drawline(w, x, color):
    if x.shape[1] == 2:
        plt.plot([np.min(x[:, 0]), np.max(x[:, 0])],
                 [np.min(x[:, 0]) * w[0, 0] + w[1, 0], np.max(x[:, 0]) * w[0, 0] +  w[1, 0]], color)

def loss(w, x, y):
    my_y = np.dot(x, w)
    diff = (y - my_y).reshape(y.shape[0],)
    res = np.dot(diff, diff) / y.shape[0]
    return res

def grad(w, x, y):
    res = (- 2 * np.dot(x.T, (y - np.dot(x, w))))/ y.shape[0]
    return res

def one_func_loss(w, x, y, i):
    my_y = np.dot(x[i], w)
    diff = (y[i] - my_y) #.reshape(my_y.shape[0],)
    res = np.dot(diff, diff)
    return res

def one_func_grad(w, x, y, i):
    x_loc = x[i]
    my_y = np.dot(x_loc, w)
    res = ((-2 * (y[i] - my_y)) * x_loc).reshape(w.shape)
    return res

def batch_grad(w, x, y, i0, i1):
    x_loc = x[i0:i1]
    y_loc = y[i0:i1]
    res = (- 2 * np.dot(x_loc.T, (y_loc - np.dot(x_loc, w)))) / y_loc.shape[0]
    return res

def batch_loss(w, x, y, i0, i1):
    x_loc = x[i0:i1]
    y_loc = y[i0:i1]
    my_y = np.dot(x_loc, w)
    diff = (y_loc - my_y).reshape(y_loc.shape[0],)
    res = np.dot(diff, diff) / y_loc.shape[0]
    return res

#Generating the data
seed(random_seed_w)
w = np.random.normal(mu, sigma1, (dim + 1, 1))
seed(random_seed_x)
x = np.concatenate((np.random.rand(data_size, dim) * x_interval, np.ones((data_size, 1))), axis = 1)
seed(random_seed_noise)
y = np.dot(x, w) + (np.random.normal(mu, sigma2, (data_size, 1)))

#Solving explicitly
# solution = np.linalg.solve((np.dot(x.T, x)), (np.dot(x.T, y)))
# true_loss = loss(solution, x, y)

#Generating the starting point
seed (random_seed_w0)
w0 = np.random.normal(mu, sigma1, (dim + 1, 1))
#w0 = w0.reshape((w0.shape[0], ))

opt_res = op.minimize(fun=lambda w: loss(w.reshape((w.size, 1)), x, y), x0=w0, tol=1e-14)
true_loss = opt_res['fun']


if __name__ == "__main__":
    print("\nTrue solution: ", solution)
    print("Starting point: ", w0)
    #Plotting

    # plt.clf()
    # plotdata(x, y)
    # drawline(solution, x, "-k")
    # plt.show()
    #Testing
    #print("Gradient: ", one_func_loss(w0, x, y, 5))