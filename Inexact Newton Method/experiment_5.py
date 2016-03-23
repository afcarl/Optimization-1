import numpy as np
import time
from matplotlib import pyplot as plt
from matplotlib import rc
from scipy.optimize import minimize
from scipy.linalg import orth
from scipy.sparse import csr_matrix
import matplotlib2tikz as tikz
from sklearn.datasets import load_svmlight_file

from optim import cg, plot_performance
from lossfuncs import logreg, logreg_hessvec

# def oracle(w):
#     return logreg(w, Z_mat, gamma, hess=False)
#
# def hess_vec(w, d):
#     return logreg_hessvec(w, d, Z_mat, gamma)


# w_opt_hfn, hist_hfn = hfn(oracle, w0, hess_vec, disp=False, maxiter=50, tol=1e-5)
# plot_performance(hist_hfn['f'], optimal_value, 'r', "Inexact Newton", time_vec=None)

dim = 100
num_clusters = 50
np.random.seed(0)
Q = np.random.rand(dim, dim)
Q = orth(Q)
s = np.array(list(range(1, 101, 2)) * 2, dtype=float)
s += np.random.normal(0, 0.001, s.shape)
A = (Q.dot(np.diag(s))).dot(Q.T)
np.random.seed(1)
b = np.random.rand(dim, 1)
def matvec(x):
    return A.dot(x)
# print(np.linalg.solve(A, b))
ans, hist = cg(matvec, b, np.zeros(b.shape), disp=False, tol=1e-10, maxiter=100)
plot_performance(hist['norm_r'], 0, 'b', "50 кластеров")

s = np.array(list(range(1, 11)) * 10, dtype=float)
s += np.random.normal(0, 0.001, s.shape)
A = (Q.dot(np.diag(s))).dot(Q.T)
np.random.seed(1)
b = np.random.rand(dim, 1)
def matvec(x):
    return A.dot(x)

# print(np.linalg.solve(A, b))
ans, hist = cg(matvec, b, np.zeros(b.shape), disp=False, tol=1e-10, maxiter=100)
plot_performance(hist['norm_r'], 0, 'r', "10 кластеров")

plt.ylabel(r'$\log||Ax_k - b||$')
plt.xlabel("Номер итерации")

title = "Метод Сопряженных Градиентов"
# title += ", D = " + str(dim + 1) + ", N = " + str(data_size) + "."
plt.title(title)
plt.legend()
file_name = "Plots/cg_"
file_name += str(dim) + "_" + str(num_clusters)
file_name += ".tikz"
tikz.save(file_name)
plt.show()