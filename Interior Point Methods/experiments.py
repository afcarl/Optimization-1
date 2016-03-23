import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_svmlight_file
import matplotlib2tikz as tikz

from l1linreg import barrier, pd, prox

def get_fun_lst_mu(x, t, mu_lst, reg_coef):
    def loss(w):
        return np.linalg.norm(t - x.dot(w))**2/2 + reg_coef * np.linalg.norm(w, 1)
    fun_lst = []
    n, d = x.shape
    x_small = x[:d, :]
    x_small_inv = np.linalg.inv(x_small)
    for mu in mu_lst:
        w = x_small_inv.dot((t + mu)[:d, :])
        fun_lst.append(loss(w))
    return fun_lst

def get_fun_lst(x, t, w_lst, reg_coef):
    def loss(w):
        return np.linalg.norm(t - x.dot(w))**2/2 + reg_coef * np.linalg.norm(w, 1)
    return [loss(w) for w in w_lst]


def plot_performance(val_vec, true_val, color, label, log_scale=True, time_vec=None):
    y_vec = [val - true_val for val in val_vec if val - true_val > np.exp(-14)]
    if len(y_vec) < len(val_vec):
        y_vec += [np.exp(-14)]
    if log_scale:
        y_vec = [np.log(y) for y in y_vec]
    x_vec = time_vec
    if x_vec is None:
        x_vec = range(len(y_vec))
    x_vec = x_vec[:len(y_vec)]
    plt.plot(x_vec, y_vec, color, label=label)


# x_d, y_d = load_svmlight_file('../../../DataSets/Regression/abalone(4177, 8).txt')
# data_name = 'abalone'
# x_d, y_d = load_svmlight_file('../../../DataSets/Regression/bodyfat(252, 14).txt')
# data_name = 'bodyfat'
x_d, y_d = load_svmlight_file('../../../DataSets/Regression/cpusmall(8192, 12).txt')
data_name = 'cpusmall'
reg_coef = 0.1
x = x_d.toarray()
y = y_d.reshape((y_d.size, 1))
n, d = x.shape

print_iter = 20
# sigma_1, sigma_2 = 1., 0.05
# np.random.seed(d)
# x = np.random.rand(n, d)
# w = np.random.normal(scale=sigma_1, size=(d, 1))
# y = x.dot(w)
# y += np.random.normal(scale=sigma_2, size=y.shape)

barrier_w_lst, pd_mu_lst, prox_w_lst = [], [], []
barrier_time_lst, pd_time_lst, prox_time_lst = [], [], []

print('Barrier')
barrier(x, y, reg_coef, display=False, w_list=barrier_w_lst, time_list=barrier_time_lst, tol_gap=1e-8, max_iter=30)
print('Proximal')
prox(x, y, reg_coef, display=False, w_list=prox_w_lst, time_list=prox_time_lst, max_iter=2000, tol_gap=1e-8)
print('Primal-Dual')
pd(x, y, reg_coef, display=True, mu_list=pd_mu_lst, time_list=pd_time_lst, tol_gap=1e-8)
print('Optimization finished')

barrier_fun_lst = get_fun_lst(x, y, barrier_w_lst, reg_coef)
prox_fun_lst = get_fun_lst(x, y, prox_w_lst, reg_coef)
pd_fun_lst = get_fun_lst_mu(x, y, pd_mu_lst, reg_coef)

opt_val = min([barrier_fun_lst[-1], prox_fun_lst[-1], pd_fun_lst[-1]])

# plot_performance(barrier_fun_lst[:print_iter], opt_val, '-r', label='Primal')
# plot_performance(prox_fun_lst[:print_iter], opt_val, '-b', label='Proximal')
# plot_performance(pd_fun_lst[:print_iter], opt_val, '-g', label='Primal-Dual')
# plt.title(data_name+', N:'+str(n)+', D:'+str(d))
# plt.ylabel(r'$\log(f_k - f_*)$')
# # plt.xlabel("Время (сек)")
# plt.xlabel("Номер итерации")
# title = "Набор данных " + data_name
# plt.title(title)
# plt.legend()
# file_name = "Plots/"
# file_name += data_name
# file_name += ".tikz"
# tikz.save(file_name)
# plt.show()

plot_performance(barrier_fun_lst, opt_val, '-r', label='Primal', time_vec=barrier_time_lst)
plot_performance(prox_fun_lst, opt_val, '-b', label='Proximal', time_vec=prox_time_lst)
plot_performance(pd_fun_lst, opt_val, '-g', label='Primal-Dual', time_vec=pd_time_lst)
plt.title(data_name+', N:'+str(n)+', D:'+str(d))
plt.ylabel(r'$\log(f_k - f_*)$')
plt.xlabel("Время (сек)")
# plt.xlabel("Номер итерации")
title = "Набор данных " + data_name
plt.title(title)
plt.legend()
file_name = "Plots/"
file_name += data_name
file_name += "_t.tikz"
tikz.save(file_name)
plt.show()