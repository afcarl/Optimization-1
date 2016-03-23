import numpy as np
import time
from matplotlib import pyplot as plt
from matplotlib import rc
from scipy.optimize import minimize
from scipy.linalg import orth
from scipy.sparse import csr_matrix, diags
import matplotlib2tikz as tikz
from sklearn.datasets import load_svmlight_file

from optim import newton, plot_performance, hfn
from lossfuncs import logreg, logreg_hessvec

# Parameters
gamma = 1e-2  # Regularization constant
dim = 4  # D - 1
data_size = 1000  # N
mu, sigma1 = 0, 0.5 # Parameters for generating the data and the starting point
x_interval = 1  # Data scaling parameter
sparse = True  # Boolean, indicating weather or not we use sparse data
real_data = False  # Boolean, indicating weather or not we use real data

# Generating the data
if real_data:
    x_d, y_d = load_svmlight_file('../../../DataSets/Classification/fourclass_scale.txt')
    data_name = 'Fourclass'
    true_loss = 0
    true_loss = 0 # 2.3828995645457518

    x = np.concatenate((x_d.toarray(), np.ones((x_d.shape[0], 1))), axis=1)
    y = y_d.reshape((y_d.size, 1))
    dim = x.shape[1] - 1
    data_size = y.size
    print(data_size, dim)
    y[y != 1] = -1
    w = np.zeros((dim+1, 1))
else:
    np.random.seed(0)
    w = np.random.normal(mu, sigma1, (dim + 1, 1))
    np.random.seed(1)
    x = np.concatenate((np.random.rand(data_size, dim) * x_interval - x_interval / 2, np.ones((data_size, 1))), axis=1)
    y = np.sign(np.dot(x, w))

#  Generating the starting point
np.random.seed(2)
w0 = np.random.normal(mu, sigma1, (dim + 1, 1))

if sparse:
    indices = np.random.randint(data_size, size=500)
    x[indices, 0] = 0
    indices = np.random.randint(data_size, size=500)
    x[indices, 1] = 0
    x = csr_matrix(x)
    Z_mat = -x.T.dot(diags([y[:, 0].tolist()], [0]))
    Z_mat = Z_mat.T.tocsr()
else:
    w0 = w + np.random.normal(np.zeros(w.shape), 0.01, w.shape)
    Z_mat = - y * x


def oracle_hess(w):
    return logreg(w, Z_mat, gamma, hess=True)


def oracle(w):
    return logreg(w, Z_mat, gamma, hess=False)


def hessvec(x, d):
    return logreg_hessvec(x, d, Z_mat, gamma)


def fun(w):
    f, g = logreg(w.reshape((w.shape[0], 1)), Z_mat, gamma, hess=False)
    return f, g[:, 0]


# def grad(w):
    # return (logreg(w.reshape((w.shape[0], 1)), Z_mat, gamma, hess=True)[1])[:, 0]

optimal_value = (minimize(fun, w0[:, 0], method="CG",
                          options={'disp': False, 'maxiter': 100}, tol=1e-5, jac=True))['fun']

print("Optimal Value Found")
# Newton method
w_opt_newton, hist_newton = newton(oracle_hess, w0, disp=False, maxiter=100, tol=1e-10)
w_opt_hfn, hist_hfn = hfn(oracle, w0, hessvec, disp=False, maxiter=200, tol=1e-7)

# optimal_value = hist_newton['f'][-1]

# Plotting
plot_performance(hist_newton['f'], optimal_value, 'b', "Метод Ньютона", time_vec=hist_newton['elaps'])
plot_performance(hist_hfn['f'], optimal_value, 'r', "Неточный Метод\n Ньютона", time_vec=hist_hfn['elaps'])
plt.ylabel(r'$\log(f_k - f_*)$')
plt.xlabel("Время (сек)")
if real_data:
    title = "Набор данных " + data_name
else:
    title = "Сгенерированные данные"
title += ", D = " + str(dim + 1) + ", N = " + str(data_size) + "."
plt.title(title)
plt.legend()
file_name = "Plots/inexact_newton_"
if real_data:
    file_name += data_name
if sparse:
    file_name += "sparse_"
if not real_data:
    file_name += str(dim+1) + "_" + str(data_size)
file_name += ".tikz"
# tikz.save(file_name)
plt.show()
