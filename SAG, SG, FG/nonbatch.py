"""In this module the FG, SG and SAG methods are implemented. SG and SAG are implemented with no batches"""
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import scipy.optimize as op
from matplotlib.legend_handler import HandlerLine2D
import time
from sklearn.datasets import load_svmlight_file
from pylab import *

#Importing the problem
from linreg import w0, x, y, true_loss, grad, loss, one_func_grad, one_func_loss, drawline, plotdata

#parameters
print_freq = 1 #the printing frequency
max_iter = 40 #maximum iteration number
stop_tol = 1e-5 #stop tolerance
graphics = 1 #1 for iterations, 0 for time
rule = 0 #0 is Armiho, 1 is Wolfe
random_matr_seed = 16

class problem:
    """ This is a class, containing the information about the problem """
    def __init__(self, data_x, data_y, loss_function, gradient, one_func_loss, one_func_grad, starting_point, \
                 solution_loss):
        self.x, self.y, self.w0, self.true_loss = data_x, data_y, starting_point, solution_loss
        self.loss, self.grad, self.one_loss, self.one_grad = loss_function, gradient, one_func_loss, one_func_grad

def plot_performance(time_vec, point_vec, prb, mode, lbl, color, freq):
    """ Performance plotting function """
    loss = prb.loss
    true_loss = prb.true_loss
    x, y = prb.x, prb.y
    if mode: #1 => iterations, 0 => time
        plt.plot(range(0, len(point_vec)*freq, freq), [np.log10(np.abs(loss(elem, x, y) - true_loss)) for\
                                                              elem in point_vec], color, label=lbl)
    else:
        plt.plot(time_vec, [np.log10(np.abs(loss(elem, x, y) - true_loss)) for\
                                                              elem in point_vec], color, label=lbl)
    plt.legend()

def stoch_average_gradient(prb, stop_tol, max_iter, freq):
    #Retrieving the parameters
    x, y, true_loss, w0 = prb.x, prb.y, prb.true_loss, prb.w0
    loss, one_func_loss, one_func_grad = prb.loss, prb.one_loss, prb.one_grad

    #Parameters
    update_rate = 10
    l = 10
    eps = 0.5

    def stoch_grad(w, x, y, i, gradient):
        local_grad = one_func_grad(w, x, y, i)
        gradient += (local_grad - grad_matrix[i].reshape(w.shape)) / y.size
        grad_matrix[i] = local_grad.reshape(w.shape[0], )
        return gradient

    w = w0
    # Resetting counters
    start = time.clock()
    time_vec = []
    iteration_counter = 0
    point_vec = []

    #Setting the required variables
    grad_matrix = np.zeros(x.shape)
    current_loss = loss(w, x, y)
    seed(random_matr_seed)
    random_matr = np.random.random_integers(0, x.shape[0]-1, (update_rate * y.size,))
    i = random_matr[iteration_counter % (y.size * update_rate)]
    gradient = np.zeros(w.shape)
    gradient = stoch_grad(w, x, y, i, gradient)

    while (iteration_counter < max_iter * y.size): #((np.linalg.norm(gradient) > stop_tol) or (iteration_counter < 2 * x.shape[0])) and \
        if iteration_counter % (x.shape[0] * freq) == 0:
            point_vec.append(w)
            time_vec.append(time.clock() - start)
            print ("SAG Iteration ", iteration_counter, ": ", np.linalg.norm(gradient))

        i = random_matr[iteration_counter % (y.size * update_rate)]
        if (iteration_counter % (y.size * update_rate) == 0):
            seed(iteration_counter)
            random_matr = np.random.random_integers(0, x.shape[0]-1, (update_rate * y.size,))
        cur_one_func_grad = one_func_grad(w, x, y, i)
        cur_one_func_loss = one_func_loss(w, x, y, i)
        l *= np.power(2, - 1 / x.shape[0])
        w_new = w - cur_one_func_grad / l
        while (one_func_loss(w_new, x, y, i)) > \
                    cur_one_func_loss - eps * np.dot(cur_one_func_grad.T, cur_one_func_grad) / l:
            l *= 2
            w_new = w - cur_one_func_grad / l

        step = 1 / l
        w = w - step * gradient
        iteration_counter += 1
        gradient = stoch_grad(w, x, y, i, gradient)
    print ("SAG Iteration ", iteration_counter, ": ", np.linalg.norm(gradient))
    point_vec.append(w)
    time_vec.append(time.clock() - start)
    return (point_vec, time_vec)

def stoch_gradient_descent(prb, stop_tol, max_iter, freq):
    #Parameters
    update_rate = 5
    step0 = 0.1
    gamma = 0.5 #used in the step size rule

    #Retrieving parameters
    x, y, true_loss, w0 = prb.x, prb.y, prb.true_loss, prb.w0
    one_func_grad = prb.one_grad

    #Resetting the counters
    w = w0
    start = time.clock()
    time_vec = []
    iteration_counter = 0
    point_vec = []
    current_loss = loss(w, x, y)
    #last_grads_matrix = np.ones(x.shape)
    random_matr = np.random.random_integers(0, x.shape[0]-1, (update_rate * y.size,))

    while (iteration_counter < max_iter * y.size): #(np.linalg.norm(np.sum(last_grads_matrix, axis = 0)) / y.size > stop_tol) and \
        if iteration_counter % (x.shape[0] * freq) == 0:
            point_vec.append(w)
            time_vec.append(time.clock() - start)
            print("SG Iteration ", iteration_counter)#, ": ",\
            #        np.linalg.norm(np.sum(last_grads_matrix, axis = 0))/ y.size)

        i = random_matr[iteration_counter % (y.size * update_rate)]
        if (iteration_counter % (y.size * update_rate) == 0):
            random_matr = np.random.random_integers(0, x.shape[0]-1, (update_rate * y.size,))
        gradient = one_func_grad(w, x, y, i)
        # last_grads_matrix[iteration_counter % y.size] = gradient
        step = step0 / np.power(iteration_counter + 1, gamma)
        w = w - step * gradient
        iteration_counter += 1
    point_vec.append(w)
    time_vec.append(time.clock() - start)
    return (point_vec, time_vec)

def full_gradient_descent(prb, stop_tol, max_iter, freq, stepsize_rule):
    #Retrieving the parameters
    x = prb.x
    y = prb.y
    w0 = prb.w0
    loss = prb.loss
    grad = prb.grad

    #Resetting the counters
    w = w0
    iteration_counter = 0
    point_vec = []
    start = time.clock()
    time_vec = []

    #Wolf rule for linear search
    w_eps1, w_eps2 = 0.0001, 0.1
    w_theta1, w_theta2 = 2, 0.5
    def wolf(w, step):
        a_up, a_down = 0, 0
        a = step
        current_loss = loss(w, x, y)
        gradient = grad(w, x, y)
        w_new = w - a * gradient
        cond1 = (loss(w_new, x, y) > current_loss + w_eps1 * a * np.square(np.linalg.norm(gradient)))
        cond2 = (-np.dot(grad(w_new, x, y).T, gradient) < - w_eps2 * np.square(np.linalg.norm(gradient)))
        while cond1 or cond2:
            if cond1:
                a_up = a
            elif cond2:
                a_down = a
                if a_up == 0:
                    a = a_down * w_theta1
                    w_new = w - a * gradient
                    cond1 = (loss(w_new, x, y) > current_loss + w_eps1 * a * np.square(np.linalg.norm(gradient)))
                    cond2 = (-np.dot(grad(w_new, x, y).T, gradient) < - w_eps2 * np.square(np.linalg.norm(gradient)))
                    continue
            a = a_up * w_theta2 + a_down *(1 - w_theta2)
            w_new = w - a * gradient
            cond1 = (loss(w_new, x, y) > current_loss + w_eps1 * a * np.square(np.linalg.norm(gradient)))
            cond2 = (-np.dot(grad(w_new, x, y).T, gradient) < - w_eps2 * np.square(np.linalg.norm(gradient)))
        return a

    #Armiho rule for linear search
    a_eps, a_theta = 0.0001, 0.5
    def armiho(w, stp):
        step = stp
        step /= a_theta
        current_loss = loss(w, x, y)
        gradient = grad(w, x, y)
        w_new = w - step * gradient
        while loss(w_new, x, y) > current_loss - a_eps * step * np.dot(gradient.T, gradient):
            step *= a_theta
            w_new = w - step * gradient
        return step

    if stepsize_rule == 1:
        step_rule = wolf
    else:
        step_rule = armiho

    step = 1
    gradient = grad(w, x, y)
    while (np.linalg.norm(gradient) > stop_tol) and (iteration_counter < max_iter):
        if iteration_counter % freq == 0:
            point_vec.append(w)
            time_vec.append(time.clock() - start)
            print ("FG Iteration ", iteration_counter, ": ", np.linalg.norm(gradient))
        step = step_rule(w, step)
        w_new = w - step * gradient
        w = w_new
        iteration_counter += 1
        gradient = grad(w, x, y)

    point_vec.append(w)
    time_vec.append(time.clock() - start)
    return (point_vec, time_vec)

def miso(prb, max_iter, freq):
    #Parameters
    update_rate = 5
    l = 1 #  the initial lipshitz constant
    random_matr_seed = 16

    #Retrieving parameters
    x, y, true_loss, w0 = prb.x, prb.y, prb.true_loss, prb.w0
    one_func_grad = prb.one_grad

    #Resetting the counters
    w = w0
    start = time.clock()
    time_vec = []
    iteration_counter = 0
    point_vec = []
    surrogate_matrix = np.zeros(x.shape) # the matrix, containing the minimizers of surogate functions
    surrogate_matrix += w0.T
    seed(random_matr_seed)
    random_matr = np.random.random_integers(0, x.shape[0]-1, (update_rate * y.size,))

    # for i in range(0, y.size / 20 + 1)):
    #     if
    while (iteration_counter < y.size):
        i = iteration_counter
        cur_w = w - one_func_grad(w, x, y, i) / l
        w += (cur_w - surrogate_matrix[i].reshape(cur_w.shape)) / y.size
        surrogate_matrix[i] = cur_w.reshape(cur_w.shape[0], )
        iteration_counter += 1

    while (iteration_counter < max_iter * y.size):
        if iteration_counter % (x.shape[0] * freq) == 0:
            point_vec.append(w)
            time_vec.append(time.clock() - start)
            print("MISO Iteration ", iteration_counter)

        i = random_matr[iteration_counter % (y.size * update_rate)]
        cur_w = w - one_func_grad(w, x, y, i) / l
        w = w + (cur_w - surrogate_matrix[i].reshape(cur_w.shape)) / y.size
        surrogate_matrix[i] = cur_w.reshape(cur_w.shape[0], )
        if (iteration_counter % (y.size * update_rate) == 0):
            seed(iteration_counter)
            random_matr = np.random.random_integers(0, x.shape[0]-1, (update_rate * y.size,))
        iteration_counter += 1
    point_vec.append(w)
    time_vec.append(time.clock() - start)
    return (point_vec, time_vec)

#Running
prb = problem(x, y, loss, grad, one_func_loss, one_func_grad, w0, true_loss)
(FG_point, FG_time) = full_gradient_descent(prb, stop_tol, max_iter, print_freq, rule)
(SAG_point, SAG_time) = stoch_average_gradient(prb, stop_tol, max_iter, print_freq)
(SG_point, SG_time) = stoch_gradient_descent(prb, stop_tol, max_iter,print_freq)
(MISO_point, MISO_time) = miso(prb, max_iter, print_freq)

FG_w = FG_point[len(FG_point) - 1]
SG_w = SG_point[len(SG_point) - 1]
SAG_w = SAG_point[len(SAG_point) - 1]
MISO_w = MISO_point[len(MISO_point) - 1]

plot_performance(FG_time, FG_point, prb, graphics, "FG", '-go', print_freq)
plot_performance(SAG_time, SAG_point, prb, graphics, "SAG", '-ro', print_freq)
plot_performance(SG_time, SG_point, prb, graphics, "SG", '-bo', print_freq)
plot_performance(MISO_time, MISO_point, prb, graphics, "MISO", '-yo', print_freq)
plt.show()
# plt.clf()
# plotdata(x, y)
# drawline(FG_w, x, "-g")
# #drawline(w_SG, x, "-b")
# #drawline(w_SAG, x, "-r")
# plt.show()
