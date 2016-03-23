import numpy as np
import time
from matplotlib import pyplot as plt


def plot_performance(val_vec, true_val, color, label, log_scale=True, time_vec=None):
    y_vec = [val - true_val for val in val_vec if val - true_val > np.exp(-14)]
    if log_scale:
        y_vec = [np.log(y) for y in y_vec]
    x_vec = time_vec
    if x_vec is None:
        x_vec = range(len(y_vec))
    x_vec = x_vec[:len(y_vec)]
    plt.plot(x_vec, y_vec, color, label=label)


def wolf(func, x, d, c1, c2, cur_loss=None, cur_grad=None, step=1, theta1=2, theta2=0.5):
    """
    Wolf rule
    :param func: oracle function, returning the function value, gradient and hessian
    :param x: evaluation point
    :param d: new method direction
    :param c1: Wolf rule parameter
    :param c2: Wolf rule parameter
    :param cur_loss: loss at point x
    :param cur_grad: gradient at point x
    :param step: initial step size (e.g. previous step size)
    :param theta1: Wolf rule parameter
    :param theta2: Wolf rule parameter
    :return a: step size, satisfying the Wolf condition
    :return loss_new: loss at the new point
    :return grad_new: gradient at the new point
    """
    if not(isinstance(x, np.ndarray)):
        raise TypeError("x should be a numpy.ndarray")
    if not(isinstance(d, np.ndarray)):
        raise TypeError("d should be a numpy.ndarray")
    if x.shape != d.shape:
        raise ValueError("x and d must have equal shapes")
    if not(hasattr(func, "__call__")):
        raise TypeError("func must be callable")
    if c1 < 0 or c1 > c2 or c2 > 1:
        raise ValueError("c1 and c2 must satisfy 0 < c1 < c2 < 1")

    a_up, a_down = 0, 0
    a = step
    if (cur_loss is None) or (cur_grad is None):
        loss, grad = func(x)[:2]
    else:
        loss, grad = cur_loss, cur_grad
    x_new = x + a * d
    oracle_res = func(x_new)
    loss_new, grad_new = oracle_res[:2]
    cond1 = loss_new > loss + c1 * a * np.dot(grad.T, d)
    cond2 = np.dot(grad_new.T, d) < c2 * np.dot(grad.T, d)
    while cond1 or cond2:
        if a_up != 0 and a_up < 1e-5:
            a = a_up
            break
        if cond1:
            a_up = a
        elif cond2:
            a_down = a
            if a_up == 0:
                a = a_down * theta1
                x_new = x + a * d
                oracle_res = func(x_new)
                loss_new, grad_new = oracle_res[:2]
                cond1 = loss_new > loss + c1 * a * np.dot(grad.T, d)
                cond2 = np.dot(grad_new.T, d) < c2 * np.dot(grad.T, d)
                continue
        a = a_up * theta2 + a_down * (1 - theta2)
        x_new = x + a * d
        oracle_res = func(x_new)
        loss_new, grad_new = oracle_res[:2]
        cond1 = loss_new > loss + c1 * a * np.dot(grad.T, d)
        cond2 = np.dot(grad_new.T, d) < c2 * np.dot(grad.T, d)
    return a, oracle_res


def newton(func, x0, disp=False, maxiter=500, tol=1e-5, c1=1e-4, c2=0.9):
    """
    Newton method
    :param func: oracle function, returning the function value, gradient and hessian
    :param x0: a D-dimensional vector, starting point
    :param disp: a flag, showing weather or not to display the method's progress
    :param maxiter: maximum number of iterations
    :param tol: the gradient discrepancy tolerance for the infinite norm
    :param c1: c1 constant in the Wolf condition
    :param c2: c2 constant in the Wolf condition
    :return x: a D-dimensional array — the estimation of the minimizer
    :return hist: a dictionary containing {'elaps': a list of iteration-wise times, 'f': a list of iteration-wise
    loss-function values, 'norm_g': a list of iteration-wise infinite norms of the gradient }
    """
    if not(isinstance(x0, np.ndarray)):
        raise TypeError("x0 should be a numpy.ndarray")
    if not(hasattr(func, "__call__")):
        raise TypeError("func must be callable")
    if tol <= 0:
        raise ValueError("tol must be greater than zero")

    time_list = []
    val_list = []
    grad_list = []
    x = np.copy(x0)
    start_time = time.time()
    loss, grad, hess = func(x0)
    iteration_counter = 0
    step_size = 1
    grad_norm = np.max(np.abs(grad))
    oracle_res = ()
    while iteration_counter < maxiter and grad_norm > tol:
        d = np.linalg.solve(hess, -grad)
        step_size, oracle_res = wolf(func, x, d, c1, c2, step=step_size)
        loss, grad, hess = oracle_res
        x += step_size * d
        _, _, hess = func(x)
        time_list.append(time.time() - start_time)
        val_list.append(loss)
        grad_norm = np.max(np.abs(grad))
        grad_list.append(grad_norm)
        if disp:
            print("_______________________")
            print("Iteration", iteration_counter, ":")
            print("Loss:", loss)
            print("Gradient norm:", grad_norm)
        iteration_counter += 1
    hist = {'elaps': time_list, 'f': val_list, 'norm_g': grad_list}
    return x, hist

def cg(matvec, b, x0, disp=False, tol=1e-5, maxiter=None):
    """
    Conjugate gradients' method for a function f(x) = (Ax, x)/2 - (b, x), i.e. solving Ax = b
    :param matvec(d): function, multiplying the system's matrix by the given vector d
    :param b: an n-dimensional vector, right hand part of the system
    :param x0: an n-dimensional vector, starting point
    :param disp: boolean flag, showing weather or not to show the method's progress
    :param tol: the infinite norm discrepancy tolerance
    :param maxiter: maximum number of iterations
    :return x: an n-dimensional vector, solution estimate
    :return hist: a dictionary {'norm_r': a list of iteration-wise discrepancy infinite norm}
    """
    if not hasattr(matvec, "__call__"):
        raise TypeError("matvec must be callable")
    if not isinstance(b, np.ndarray):
        raise TypeError("b must be a numpy.ndarray")
    if not isinstance(x0, np.ndarray):
        raise TypeError("x0 must be a numpy.ndarray")
    if b.shape != x0.shape:
        raise ValueError("b and x0 must have equal shapes")
    if tol <= 0:
        raise ValueError("tol must be greater than zero")

    x = np.copy(x0)
    g = matvec(x) - b
    d = -g
    u = matvec(d)
    g_norm = np.max(np.abs(g))
    iteration_counter = 0
    norm_list = [np.linalg.norm(matvec(x) - b)]
    if maxiter is None:
        maxiter = np.inf
    while g_norm > tol and iteration_counter < maxiter:
        if disp:
            print("_______________________")
            print("Iteration", iteration_counter, ":")
            print("Discrepancy norm:", norm_list[-1])
        alpha = - g.T.dot(d) / (u.T.dot(d))
        x += alpha * d
        g_new = g + alpha * u
        g_norm = np.max(np.abs(g_new))
        if g_norm <= tol:
            norm_list.append(np.linalg.norm(matvec(x) - b))
            break
        beta = g_new.T.dot(g_new) / (g.T.dot(g))
        g = g_new
        d = - g + beta * d
        u = matvec(d)

        norm_list.append(np.linalg.norm(matvec(x) - b))
        iteration_counter += 1
    hist = {'norm_r': norm_list}
    return x, hist


def hfn(func, x0, hessvec, disp=False, maxiter=500, tol=1e-5, c1=1e-4, c2=0.9):
    """
    Inexact Newton method
    :param func: oracle function, returning the function value, gradient
    :param x0: a D-dimensional vector, starting point
    :param hessvec(x, d): function, multiplying the hessian at point x by vector d
    :param disp: a flag, showing weather or not to display the method's progress
    :param maxiter: maximum number of iterations
    :param tol: the gradient discrepancy tolerance for the infinite norm
    :param c1: c1 constant in the Wolf condition
    :param c2: c2 constant in the Wolf condition
    :return x: a D-dimensional array — the estimation of the minimizer
    :return hist: a dictionary containing {'elaps': a list of iteration-wise times, 'f': a list of iteration-wise
    loss-function values, 'norm_g': a list of iteration-wise infinite norms of the gradient }
    """
    if not(isinstance(x0, np.ndarray)):
        raise TypeError("x0 should be a numpy.ndarray")
    if not(hasattr(func, "__call__")):
        raise TypeError("func must be callable")
    if not hasattr(hessvec, "__call__"):
        raise TypeError("hessvec must be callable")
    if tol <= 0:
        raise ValueError("tol must be greater than zero")

    time_list = []
    val_list = []
    grad_list = []
    x = np.copy(x0)
    start_time = time.time()
    loss, grad = func(x0)
    iteration_counter = 0
    step_size = 1
    grad_norm = np.max(np.abs(grad))
    eta = min(0.5, np.sqrt(grad_norm))
    oracle_res = ()
    while iteration_counter < maxiter and grad_norm > tol:
        d = cg(lambda d: hessvec(x, d), -grad, np.zeros(x.shape), tol=eta * grad_norm)[0]
        step_size, oracle_res = wolf(func, x, d, c1, c2, step_size)
        loss, grad = oracle_res
        eta = min(0.5, np.sqrt(grad_norm))
        x += step_size * d
        time_list.append(time.time() - start_time)
        val_list.append(loss)
        grad_norm = np.max(np.abs(grad))
        grad_list.append(grad_norm)
        if disp:
            print("_______________________")
            print("Iteration", iteration_counter, ":")
            print("Loss:", loss)
            print("Gradient norm:", grad_norm)
        iteration_counter += 1
    hist = {'elaps': time_list, 'f': val_list, 'norm_g': grad_list}
    return x, hist
