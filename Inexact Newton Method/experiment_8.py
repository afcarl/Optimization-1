import numpy as np
import time
from matplotlib import pyplot as plt
from matplotlib import rc
from scipy.optimize import minimize
from scipy.linalg import orth
from scipy.sparse import csr_matrix, diags
import matplotlib2tikz as tikz
from sklearn.datasets import load_svmlight_file

from optim import plot_performance, hfn, newton
from lossfuncs import logreg, logreg_hessvec


def write_hist(hist, elaps, f, norm_g):
    hist['elaps'].append(elaps)
    hist['f'].append(f)
    hist['norm_g'].append(norm_g)

def minimize_wrapper(func, x0, mydisp=False, **kwargs):
    hist = {'elaps':[], 'f':[], 'norm_g':[]}
    if mydisp:
        print('%9s %15s %15s' % ('elaps', 'f', 'norm_g'))

    aux = {'tstart': time.time(), 'elaps': 0}
    def callback(x):
        aux['elaps'] += time.time() - aux['tstart']
        f, g = func(x)
        norm_g = np.linalg.norm(g, np.inf)
        write_hist(hist, aux['elaps'], f, norm_g)
        if mydisp:
            print('%9s %15.6e %15.6e' % (aux['elaps'], f, norm_g))
        aux['tstart'] = time.time()
    callback(x0)
    out = minimize(func, x0, jac=True, callback=callback, **kwargs)

    return out, hist





# Parameters
gamma = 1e-2  # Regularization constant
dim = 4  # D - 1
data_size = 10000  # N
mu, sigma1 = 0, 0.5 # Parameters for generating the data and the starting point
x_interval = 1  # Data scaling parameter
sparse = True  # Boolean, indicating weather or not we use sparse data

# Generating the data
# x_d, y_d = load_svmlight_file('../../../DataSets/Classification/gisette_scale.txt')
# data_name = 'Gisette'
x_d, y_d = load_svmlight_file('../../../DataSets/Classification/leukemia.txt')
data_name = 'Leukemia'
# x_d, y_d = load_svmlight_file('../../../DataSets/Classification/real-sim.txt')
# data_name = 'Real-sim'

# x = np.concatenate((x_d.toarray(), np.ones((x_d.shape[0], 1))), axis=1)
x = x_d
y = y_d.reshape((y_d.size, 1))
print(type(x))
dim = x.shape[1] - 1
data_size = y.size
print(data_size, dim)
y[y != 1] = -1
w = np.zeros((dim+1, 1))


#  Generating the starting point
np.random.seed(2)
w0 = np.random.normal(mu, sigma1, (dim + 1, 1))

if sparse:
    print(y.shape)
    Z_mat = -x.T.dot(diags([y[:, 0].tolist()], [0]))
    Z_mat = Z_mat.T.tocsr()
    print(Z_mat.shape)
    # w0 = w
    # Z_mat = - y[:, 0] * x
    # print(Z_mat.shape)
else:
    x = x.toarray()
    w0 = w + np.random.normal(np.zeros(w.shape), 0.01, w.shape)
    Z_mat = - y * x


def oracle(w):
    return logreg(w, Z_mat, gamma, hess=False)


def oracle_hess(w):
    return logreg(w, Z_mat, gamma, hess=True)


def hessvec(x, d):
    return logreg_hessvec(x, d, Z_mat, gamma)


def fun(w):
    f, g = logreg(w.reshape((w.shape[0], 1)), Z_mat, gamma, hess=False)
    return f, g[:, 0]

print("Estimating the starting point")
optimal_value_bfgs, _ = minimize_wrapper(fun, w0, mydisp=True,
                                       method="L-BFGS-B", options={'ftol':0, 'maxiter': 1})

starting_w = optimal_value_bfgs['x'].reshape(w.shape)
w0 = starting_w

print("CG")
optimal_value_cg, hist_cg = minimize_wrapper(fun, w0, mydisp=True,
                                             method="CG", options={'maxiter': 10})

print("BFGS")
optimal_value_bfgs, hist_bfgs = minimize_wrapper(fun, w0, mydisp=True,
                                       method="L-BFGS-B", options={'ftol':0, 'maxiter': 20})

print("Inexact Newton")
w_opt_hfn, hist_hfn = hfn(oracle, w0, hessvec, disp=True, maxiter=10, tol=1e-9)

print("Newton")
w_opt_newton, hist_newton = newton(oracle_hess, w0, disp=True, maxiter=10, tol=1e-9)

# Plotting
# print(optimal_value_cg['fun'])
# print(optimal_value_bfgs['fun'])
# print((hist_hfn['f'])[-1])

optimal_value = min([optimal_value_cg['fun'], optimal_value_bfgs['fun'], (hist_hfn['f'])[-1]])
plot_performance(hist_hfn['f'], optimal_value, 'r', "Неточный Метод Ньютона", time_vec=hist_hfn['elaps'])
plot_performance(hist_newton['f'], optimal_value, 'y', "Метод Ньютона", time_vec=hist_newton['elaps'])
plot_performance(hist_bfgs['f'], optimal_value, 'b', "L-BFGS-B", time_vec=hist_bfgs['elaps'])
plot_performance(hist_cg['f'], optimal_value, 'g', "CG", time_vec=hist_cg['elaps'])
plt.ylabel(r'$\log(f_k - f_*)$')
plt.xlabel("Время (сек)")
title = "Набор данных " + data_name
plt.title(title)
plt.legend()
file_name = "Plots/inexact_newton_"
file_name += data_name
if sparse:
    file_name += "sparse_"
file_name += ".tikz"
# tikz.save(file_name)
plt.show()
