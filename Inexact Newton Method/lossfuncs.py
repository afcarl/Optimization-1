import numpy as np
from scipy.sparse import csr_matrix, diags
from matplotlib import pyplot as plt


def logreg(w, Z, regcoef, hess=False):
    """
    :param w: a D-dimensional vector at which the oracle is evaluated
    :param Z: an (N x D) matrix — numpy.ndarray or scipy.sparse.csr_matrix
    :param regcoef: regularization coefficient
    :param hess: a flag, showing weather or not to evaluate the hessian
    :return f: logistic loss function value at w
    :return g: gradient of the logistic loss function at w, a D-dimensional vector
    :return H: hessian of the logistic loss function at w, a (D x D) matrix
    """
    if not(isinstance(w, np.ndarray)):
        raise TypeError("w should be a numpy.ndarray")
    if not(isinstance(Z, np.ndarray) or isinstance(Z, csr_matrix)):
        raise TypeError("Z should be a numpy.ndarray or a csr_matrix")
    if w.shape[0] != Z.shape[1]:
        raise ValueError("w.shape[0] and Z.shape[1] mast be equal")
    if regcoef <= 0:
        raise ValueError("Regularization coefficient must be greater then 0")

    w = w.reshape((w.size, 1))
    anc_var = np.exp(-Z.dot(w))
    f = np.sum(np.log(1 + np.exp(Z.dot(w))), axis=0) + regcoef * np.linalg.norm(w)**2 / 2
    if isinstance(Z, csr_matrix):
        g = np.sum(((diags([(1 / (1 + anc_var))[:, 0].tolist()], [0])).dot(Z)).toarray(), axis=0).reshape(w.shape)
        g += regcoef * w
    else:
        g = np.sum(Z / (1 + anc_var), axis=0).reshape(w.shape) + regcoef * w
    if hess:
        if isinstance(Z, csr_matrix):
            h = diags([(anc_var / np.square(1 + anc_var))[:, 0].tolist()], [0]).dot(Z)
        else:
            h = Z * (anc_var / np.square(1 + anc_var))
        h = Z.T.dot(h) + regcoef * np.eye(w.size)
        return f, g, h
    return f, g


def logreg_hessvec(w, d, Z, regcoef):
    """
    :param w: a D-dimensional vector, the point of evaluation of the hessian
    :param d: a D-dimensional vector
    :param Z: an (N x D) matrix — numpy.ndarray or scipy.sparse.csr_matrix
    :param regcoef: regularization coefficient
    :return Hd: the dot product of the hessian of the logistic loss-function at point w and d
    """
    if not(isinstance(w, np.ndarray)):
        raise TypeError("w should be a numpy.ndarray")
    if not(isinstance(d, np.ndarray)):
        raise TypeError("d should be a numpy.ndarray")
    if not(isinstance(Z, np.ndarray) or isinstance(Z, csr_matrix)):
        raise TypeError("Z should be a numpy.ndarray or a csr_matrix")
    if w.shape[0] != Z.shape[1]:
        raise ValueError("w.shape[0] and Z.shape[1] mast be equal")
    if w.shape != d.shape:
        raise ValueError("w and d must have equal shapes")
    if regcoef <= 0:
        raise ValueError("Regularization coefficient must be greater then 0")

    anc_var = np.exp(-Z.dot(w))
    if isinstance(Z, csr_matrix):
        res = Z.dot(d)
        res = diags([(anc_var / np.square(1 + anc_var))[:, 0].tolist()], [0]).dot(res)
        res = Z.T.dot(res) + regcoef * d
    else:
        res = (Z * (anc_var / np.square(1 + anc_var))).dot(d)
        res = Z.T.dot(res) + regcoef * d
    return res


def draw_points(x, y, c1='rx', c2='bx'):
    if x.shape[1] == 3:
        loc_y = y.reshape((y.shape[0],))
        plt.plot(x[loc_y == 1, 0], x[loc_y == 1, 1], c1)
        plt.plot(x[loc_y == -1, 0], x[loc_y == -1, 1], c2)


def drawline(w, x_interval, color='g'):
    if w.size == 3:
        x0_left = -x_interval / 2
        x0_right = x_interval / 2
        x1_left = -(w[0] * x0_left + w[2]) / w[1]
        x1_right = -(w[0] * x0_right + w[2]) / w[1]
        plt.plot([x0_left, x0_right], [x1_left, x1_right], color)
        plt.axis((-x_interval / 2, x_interval / 2, -x_interval / 2, x_interval / 2))
