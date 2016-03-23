import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import scipy.optimize as op
from matplotlib.legend_handler import HandlerLine2D
import time
from sklearn.datasets import load_svmlight_file
from pylab import *

#Importing the problem
from reglinreg import w0, x, y, true_loss, grad, loss, drawline, plotdata, batch_loss, batch_grad, data_name, m, n
# from linreg import w0, x, y, true_loss, grad, loss, drawline, plotdata, batch_loss, batch_grad, data_name, m, n
# from logreg import x, y, w0, true_loss, grad, loss, batch_loss, batch_grad, data_name, m, n, test_err
# from reallog import x, y, w0, true_loss, grad, loss, batch_loss, batch_grad, data_name, m, n, test_err
# from reallin import x, y, w0, true_loss, grad, loss, batch_loss, batch_grad, data_name, m, n

#parameters
print_freq = 5 #the printing frequency
max_iter = 1000 #maximum iteration number
max_time  = 60 #maximum time
stop_tol = 1e-5 #stop tolerance
graphics = 1 #1 => iter, loss; 2 => iter, error; -1 => time, loss; -2 => time, error
batch_size = int(y.size / 10) #batch size
random_matr_seed = 2

if graphics > 0:
    max_time = inf
else:
    max_iter = inf

class problem:
    """ This is a class, containing the information about the problem """
    def __init__(self, data_x, data_y, testing_x, testing_y, loss_function, gradient, starting_point, \
                 solution_loss, batch_loss_func, batch_gradient):
        self.x, self.y, self.w0, self.true_loss = data_x, data_y, starting_point, solution_loss
        self.text_x, self.test_y = testing_x, testing_y
        self.loss, self.grad = loss_function, gradient
        self.batch_loss, self.batch_grad = batch_loss_func, batch_gradient


def plot_performance(time_vec, point_vec, prb, mode, lbl, color, freq):
    """ Performance plotting function """
    loss = prb.loss
    true_loss = prb.true_loss
    x, y = prb.x, prb.y
    x_test, y_test = prb.text_x, prb.test_y
    if mode == 1: #1 => iterations, loss; 2 => iterations, error; -1 => time, loss; -2 => time, error
        plt.plot(range(0, len(point_vec)*freq, freq), [np.log10(np.abs(loss(elem, x, y) - true_loss)) for
                                                    elem in point_vec], color, label=lbl)
    elif mode == 2:
        plt.plot(range(0, len(point_vec)*freq, freq), [test_err(elem, x_test, y_test) for
                                                    elem in point_vec], color, label=lbl)
    elif mode == -1:
        plt.plot(time_vec, [np.log10(np.abs(loss(elem, x, y) - true_loss)) for
                            elem in point_vec], color, label=lbl)
    elif mode == -2:
        plt.plot(time_vec, [test_err(elem, x_test, y_test) for
                                                    elem in point_vec], color, label=lbl)
    if mode == 1 or mode == -1:
        plt.ylabel('log10 of loss discrepancy')
    elif mode == 2 or mode == -2:
        plt.ylabel('Precision on test data set')
    if mode > 0:
        plt.xlabel('Effective passes')
    else:
        plt.xlabel('time (s)')
    plt.legend()

######################################################################################################################
######################################################################################################################

def batch_stoch_average_gradient(prb, batch_size, max_iter, max_time, freq, stepsize_rule):
    """Batch implementation of the SAG method to utilize Python vectorized computation potential"""
    print("SAG")
    #Parameters
    random_matr_seed = 16
    update_rate = 10
    l = 0.1
    eps = 0.5
    #Retrieving the parameters
    x, y, true_loss, w0 = prb.x, prb.y, prb.true_loss, prb.w0
    loss, batch_loss, batch_grad = prb.loss, prb.batch_loss, prb.batch_grad

    def stoch_grad(w, x, y, i, gradient):
        local_grad = batch_grad(w, x, y, i * batch_size, (i + 1) * batch_size)
        gradient += (local_grad - grad_matrix[i].reshape(w.shape)) / batch_num
        grad_matrix[i] = local_grad.reshape(w.shape[0], )
        return gradient

    # def grad_rule(l, i, w, gradient):
    #     cur_grad_norm = np.linalg.norm(gradient)
    #     i0 = i * batch_size
    #     i1 = (i + 1) * batch_size
    #     cur_batch_grad = batch_grad(w, x, y, i0, i1)
    #     l *= np.power(2.0, - 1 / batch_num)
    #     w_new = w - cur_batch_grad / l
    #     while cur_grad_norm < np.linalg.norm(stoch_grad(w_new, x, y, i, gradient)):
    #         l *= 2
    #         print(np.linalg.norm(stoch_grad(w_new, x, y, i, gradient)) - cur_grad_norm)
    #         w_new = w - cur_batch_grad / l
    #     return l

    def one_func_rule(l, i, w):
        i0 = i * batch_size
        i1 = (i + 1) * batch_size
        cur_batch_grad = batch_grad(w, x, y, i0, i1)
        cur_batch_loss = batch_loss(w, x, y, i0, i1)
        l *= np.power(2.0, - 1 / batch_num)
        w_new = w - cur_batch_grad / l
        while batch_loss(w_new, x, y, i0, i1) > \
                cur_batch_loss - eps * np.dot(cur_batch_grad.T, cur_batch_grad) / l:
            l *= 2
            w_new = w - cur_batch_grad / l
        return l

    # Resetting counters
    start = time.clock()
    time_vec = []
    iteration_counter = 0
    point_vec = []

    #  Setting the required variables
    w = w0
    batch_num = int(y.size / batch_size)
    if y.size % batch_size:
        batch_num += 1
    grad_matrix = np.zeros((batch_num, x.shape[1]))
    seed(random_matr_seed)
    random_matr = np.random.random_integers(0, batch_num - 1, (update_rate * batch_num,))
    i = random_matr[iteration_counter % (batch_num * update_rate)]
    gradient = np.zeros(w.shape)
    gradient = stoch_grad(w, x, y, i, gradient)
    if stepsize_rule:
        step_rule = grad_rule
    else:
        step_rule = one_func_rule

    while True:
        if iteration_counter % batch_num == 0:
            if iteration_counter % (batch_num * freq) == 0:
                point_vec.append(w)
                time_vec.append(time.clock() - start)
                # print ("Batch SAG Iteration ", iteration_counter)
            if iteration_counter >= max_iter * batch_num:
                break
            if time.clock() - start >= max_time:
                break

        i = random_matr[iteration_counter % (batch_num * update_rate-1)]
        if iteration_counter % (batch_num * update_rate) == 0:
            seed(iteration_counter)
            random_matr = np.random.random_integers(0, batch_num-1, (update_rate * batch_num,))
        l = step_rule(l, i, w)
        step = 1 / l
        w = w - step * gradient
        iteration_counter += 1
        gradient = stoch_grad(w, x, y, i, gradient)
    # print("Batch SAG Iteration ", iteration_counter, ": ")
    point_vec.append(w)
    time_vec.append(time.clock() - start)
    return(point_vec, time_vec)

######################################################################################################################
######################################################################################################################

def stoch_gradient_descent(prb, batch_size, max_iter, max_time, freq):
    print("SG")
    #Parameters
    update_rate = 5
    step0 = 0.00005
    gamma = 0.55 #used in the step size rule
    random_matr_seed = 16

    #Retrieving parameters
    x, y, true_loss, w0 = prb.x, prb.y, prb.true_loss, prb.w0
    batch_grad = prb.batch_grad

    #setting the required variables
    w = w0
    batch_num = int(y.size / batch_size)
    if y.size % batch_size:
        batch_num += 1
    seed(random_matr_seed)
    random_matr = np.random.random_integers(0, batch_num-1, (update_rate * batch_num,))

    #Resetting the counters
    time_vec = []
    iteration_counter = 0
    point_vec = []
    start = time.clock()

    while True:
        if iteration_counter % batch_num == 0:
            if iteration_counter % (batch_num * freq) == 0:
                point_vec.append(w)
                time_vec.append(time.clock() - start)
                # print("SG Iteration ", iteration_counter)
            if iteration_counter >= max_iter * batch_num:
                break
            if time.clock() - start >= max_time:
                break
            step = step0 / np.power((iteration_counter / batch_num) + 1, gamma)

        i = random_matr[iteration_counter % (batch_num * update_rate - 1)]
        if (iteration_counter % (batch_num * update_rate) == 0):
            seed(iteration_counter)
            random_matr = np.random.random_integers(0, batch_num-1, (update_rate * batch_num,))
        gradient = batch_grad(w, x, y, i * batch_size, (i + 1) * batch_size)
        w = w - step * gradient
        iteration_counter += 1
    point_vec.append(w)
    time_vec.append(time.clock() - start)
    return (point_vec, time_vec)

######################################################################################################################
######################################################################################################################

def full_gradient_descent(prb, max_iter, max_time, freq, stepsize_rule):
    print("FG")
    #Retrieving the parameters
    x = prb.x
    y = prb.y
    w0 = prb.w0
    true_loss = prb.true_loss
    loss = prb.loss
    grad = prb.grad

    #Resetting the counters
    w = w0
    iteration_counter = 0
    point_vec = []
    start = time.clock()
    time_vec = []

    #Wolf rule for linear search
    w_eps1, w_eps2 = 0.1, 0.1
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
                    cond1 = (loss(w_new, x, y) > current_loss - w_eps1 * a * np.square(np.linalg.norm(gradient)))
                    cond2 = (-np.dot(grad(w_new, x, y).T, gradient) < - w_eps2 * np.square(np.linalg.norm(gradient)))
                    continue
            a = a_up * w_theta2 + a_down *(1 - w_theta2)
            w_new = w - a * gradient
            cond1 = (loss(w_new, x, y) > current_loss + w_eps1 * a * np.square(np.linalg.norm(gradient)))
            cond2 = (-np.dot(grad(w_new, x, y).T, gradient) < - w_eps2 * np.square(np.linalg.norm(gradient)))
        return a

    #Armiho rule for linear search
    a_eps, a_theta = 0.1, 0.5
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
    while (iteration_counter < max_iter) and (time.clock() - start < max_time):
        if iteration_counter % freq == 0:
            point_vec.append(w)
            time_vec.append(time.clock() - start)
            # print ("FG Iteration ", iteration_counter)
        step = step_rule(w, step)
        w_new = w - step * gradient
        w = w_new
        iteration_counter += 1
        gradient = grad(w, x, y)
        # print(np.linalg.norm(gradient), step)
    point_vec.append(w)
    time_vec.append(time.clock() - start)
    return (point_vec, time_vec)

######################################################################################################################
######################################################################################################################

def miso1(prb, batch_size, max_iter, max_time, freq):
    print("MISO")
    #  Parameters
    update_rate = 5
    random_matr_seed = 16

    #  Retrieving parameters
    x, y, true_loss, w0 = prb.x, prb.y, prb.true_loss, prb.w0
    batch_grad = prb.batch_grad
    loss = prb.loss

    #  Setting the required variables
    w = 0
    w += w0
    k_list = range(0, 10)
    batch_num = int(y.size / batch_size)
    if y.size % batch_size:
        batch_num += 1
    surrogate_matrix = np.zeros((batch_num, x.shape[1])) # the matrix, containing the minimizers of surogate functions
    surrogate_matrix += w0.T
    seed(random_matr_seed)
    random_matr = np.random.random_integers(0, batch_num-1, (update_rate * batch_num,))

    #  Resetting the counters
    start = time.clock()
    time_vec = []
    iteration_counter = 0
    point_vec = []

    #l_0 = real(max(np.linalg.eig(np.dot(x, x.T))[0]))
    l_0 = np.linalg.norm(x)
    l_0 *= l_0
    l_0 /= y.size
    #determining the best l
    small_data_size = int(batch_num / 20)
    if batch_num % 20:
        small_data_size += 1
    l_loss = [0] * len(k_list)
    # small_data_size = int(batch_num)
    j = 0
    for k in k_list:
        small_surrogate_matrix = np.zeros((small_data_size, x.shape[1]))
        small_surrogate_matrix += w0.T
        w = 0
        w += w0
        l = l_0 * np.power(2.0, -k)
        for i in range(0, small_data_size):
            cur_w = w - batch_grad(w, x, y, i * batch_size, (i + 1) * batch_size) / l
            w += (cur_w - small_surrogate_matrix[i].reshape(cur_w.shape)) / small_data_size
            small_surrogate_matrix[i] = cur_w.reshape(cur_w.shape[0], )
        l_loss[j] = loss(w, x[0:small_data_size * batch_size], y[0:small_data_size * batch_size])
        j += 1
    k = k_list[l_loss.index(min(l_loss))]
    l = l_0 * np.power(2.0, -k)
    w = 0
    w += w0
    point_vec.append(w0)
    time_vec.append(time.clock() - start)
    print(l)

    while iteration_counter < batch_num:
        i = iteration_counter
        cur_w = w - batch_grad(w, x, y, i * batch_size, (i + 1) * batch_size) / l
        w += (cur_w - surrogate_matrix[i].reshape(cur_w.shape)) / batch_num
        surrogate_matrix[i] = cur_w.reshape(cur_w.shape[0], )
        iteration_counter += 1

    while iteration_counter < max_iter * batch_num:
        if iteration_counter % batch_num == 0:
            if iteration_counter % (batch_num * freq) == 0:
                point_vec.append(w)
                time_vec.append(time.clock() - start)
                # print("MISO Iteration ", iteration_counter)
            if iteration_counter >= max_iter * batch_num:
                break
            if time.clock() - start >= max_time:
                break

        i = random_matr[iteration_counter % (batch_num * update_rate - 1)]
        cur_w = w - batch_grad(w, x, y, i * batch_size, (i + 1) * batch_size) / l
        w = w + (cur_w - surrogate_matrix[i].reshape(cur_w.shape)) / batch_num
        surrogate_matrix[i] = cur_w.reshape(cur_w.shape[0], )
        if (iteration_counter % (batch_num * update_rate) == 0):
            seed(iteration_counter)
            random_matr = np.random.random_integers(0, batch_num - 1, (update_rate * batch_num,))
        iteration_counter += 1
    point_vec.append(w)
    time_vec.append(time.clock() - start)
    return (point_vec, time_vec)

#Running
if (graphics == 2) or (graphics == -2):
    x_test = x[int(4/5 * y.size):y.size]
    y_test = y[int(4/5 * y.size):y.size]
    x = x[0:int(4/5 * y.size)]
    y = y[0:int(4/5 * y.size)]
else:
    x_test = []
    y_test = []
prb = problem(x, y, x_test, y_test, loss, grad, w0, true_loss, batch_loss, batch_grad)
(a_FG_point, a_FG_time) = full_gradient_descent(prb, max_iter, max_time, print_freq, 0)
(w_FG_point, w_FG_time) = full_gradient_descent(prb, max_iter, max_time, print_freq, 1)
(f_SAG_point, f_SAG_time) = batch_stoch_average_gradient(prb, batch_size, max_iter, max_time, print_freq, 0)
(SG_point, SG_time) = stoch_gradient_descent(prb, batch_size, max_iter, max_time, print_freq)
(MISO_point, MISO_time) = miso1(prb, batch_size, max_iter, max_time, print_freq)

a_FG_w = a_FG_point[len(a_FG_point) - 1]
SG_w = SG_point[len(SG_point) - 1]
SAG_w = f_SAG_point[len(f_SAG_point) - 1]
MISO_w = MISO_point[len(MISO_point) - 1]

# print(FG_w, SG_w, SAG_w)

plot_performance(a_FG_time, a_FG_point, prb, graphics, "A-FG", '-go', print_freq)
plot_performance(w_FG_time, w_FG_point, prb, graphics, "W-FG", '-gs', print_freq)
plot_performance(SG_time, SG_point, prb, graphics, "SG", '-bo', print_freq)
# plot_performance(g_SAG_time, g_SAG_point, prb, graphics, "G-SAG", '-ro', print_freq)
plot_performance(f_SAG_time, f_SAG_point, prb, graphics, "SAG", '-ro', print_freq)
plot_performance(MISO_time, MISO_point, prb, graphics, "MISO", '-yo', print_freq)
if graphics > 0:
    plt.title("Comparison with respect to effective passes\n on %s data: m = %d, n = %d" % (data_name, m, n))
else:
    plt.title("Comparison with respect to time\n on %s data : m = %d, n = %d" % (data_name, m, n))
plt.show()
# plt.clf()
# plotdata(x, y)
# drawline(a_FG_w, x, "-g")
# drawline(SG_w, x, "-b")
# drawline(SAG_w, x, "-r")
# plt.show()
