import numpy as np
import time


def armiho(oracle, w, d, step_seq, a_eps=0.05):
    """
    A procedure to choose the step-size, using the Armiho rule
    :param oracle: oracle function, must take w as an argument and return the corresponding loss and gradient
    :param w: evaluation point
    :param d: direction of optimization
    :param step_seq: a generator for the sequence of step sizes
    :param a_eps: Armiho rule paramerer
    :return: step size
    """
    step = next(step_seq)
    current_loss, gradient = oracle(w)[:2]
    w_new = w + step * d
    new_loss = oracle(w_new)[:1]
    while np.isinf(new_loss):
            step = next(step_seq)
            w_new = w + step * d
            new_loss = oracle(w_new)[:1]
    while new_loss > current_loss + a_eps * step * np.dot(gradient.T, d):
        step = next(step_seq)
        w_new = w + step * d
        new_loss = oracle(w_new)[:1]
        while np.isinf(new_loss):
            step = next(step_seq)
            w_new = w + step * d
            new_loss = oracle(w_new)[:1]
    return step

def barrier(X, t, reg_coef, max_iter=100, max_inner_iter=20, tol_gap=1e-5, tol_center=1e-10,
            tau_params=np.array([1, 10]), bt_params=np.array([1e-4, 0.8]), display=0, w_list=None, time_list=None):
    """
    Primal barrier method
    :param X: (NxD) numpy.array, training set
    :param t: (N,) numpy.array, target values
    :param reg_coef: regularization coefficient
    :param max_iter: maximum iteration number
    :param max_inner_iter: maximum number of inner iterations for centring
    :param tol_gap: tolerance for gap between the values of the primal and dual problem
    :param tol_center: centring tolerance
    :param tau_params: numpy.array([tau_start, nu]), strategy for centring parameter tau_n = tau_start * nu
    :param bt_params: numpy.array([c_1, beta]), backtracking strategy parameters
    :param display: a number, showing weather or not to display progress iteration-wise
    :param w_list: a list for saving the method's progress
    :param time_list: a list for saving the iteration-wise times
    :return w: (D,) numpy.array, optimal weights
    """
    if not isinstance(X, np.ndarray) or not isinstance(t, np.ndarray) or not isinstance(tau_params, np.ndarray) or not \
            isinstance(bt_params, np.ndarray):
        raise TypeError("X, t, bt_params and tau_params must be numpy arrays")
    if X.shape[0] != t.shape[0]:
        raise ValueError("X.shape[0] != t.shape[0]")
    if len(t.shape) == 1:
        t = t[:, None]
    elif t.shape[1] != 1.:
        raise ValueError("t.shape must be of the form (N,) or (N, 1)")

    def step_size(w, d):
        """
        A procedure to choose the step-size, using the Armiho rule
        :param oracle: oracle function, must take w as an argument and return the corresponding loss and gradient
        :param w: evaluation point
        :param d: direction of optimization
        :param step_seq: a generator for the sequence of step sizes
        :param a_eps: Armiho rule paramerer
        :return: step size
        """
        a_eps = bt_params[0]
        step_seq = bt_sequence()
        step = next(step_seq)
        current_loss, gradient = oracle(w, tau)[:2]
        w_new = w + step * d
        new_loss = oracle(w_new, tau, fun_only=True)
        while np.isinf(new_loss):
                step = next(step_seq)
                w_new = w + step * d
                new_loss = oracle(w_new, tau, fun_only=True)
        while new_loss > current_loss + a_eps * step * np.dot(gradient.T, d):
            step = next(step_seq)
            w_new = w + step * d
            new_loss = oracle(w_new, tau, fun_only=True)
            while np.isinf(new_loss):
                step = next(step_seq)
                w_new = w + step * d
                new_loss = oracle(w_new, tau,  fun_only=True)
        return step

    def tau_sequence():
        """
        Generator for the sequence of centering parameters tau
        :return: tau
        """
        res = tau_params[0]
        while True:
            yield res
            res *= tau_params[1]

    def bt_sequence():
        """
        Generator for the backtracking
        :return: step size
        """
        res = 1
        while True:
            yield res
            res *= bt_params[1]

    def oracle(point, tau, fun_only=False):
        """
        oracle function for l1-regularized logistic regression in it's smooth representation
        :param w: evaluation point
        :param u: evaluation point
        :param tau: centring parameter
        :param fun_only: a boolean, showing if the hessian and gradient are needed
        :return: tuple (loss, gradient, hessian)
        """
        w = point[:d, :]
        u = point[d:, :]
        if np.any(np.abs(w) >= np.abs(u)):
            return np.inf
        if np.any(u <= 0):
            return np.inf
        loss = np.linalg.norm(t - X.dot(w))**2 / 2 + reg_coef * np.sum(u) - np.sum(np.log(u - w) + np.log(u + w)) / tau
        if fun_only:
            return loss
        anc_mat = X.T.dot(X)
        grad = -X.T.dot(t) + anc_mat.dot(w) + (1 / (u - w) - 1/(u + w)) / tau
        grad = np.vstack((grad, np.ones(u.shape) * reg_coef - (1 / (u - w) + 1/(u + w)) / tau))
        A = np.diag((1 /(u - w)**2 + 1 /(u + w)**2)[:, 0]) / tau
        B = np.diag((1 /(u + w)**2 - 1 /(u - w)**2)[:, 0]) / tau
        hess = np.hstack((anc_mat + A, B))
        hess = np.vstack((hess, np.hstack((B, A))))
        return (loss, grad, hess)

    def newton_direction(hessian, gradient):
        """
        Compute the newton direction
        :param hessian: hessian
        :param gradient: gradient
        :return: hessian^(-1) * grad
        """
        A, B = hessian[d:, d:], hessian[d:, :d]
        dw, du = gradient[:d, :], gradient[d:, :]
        A_inv, B_inv = np.diag(1 / np.diag(A)), np.diag(1 / np.diag(B))
        S = X.T.dot(X) + A - B.dot(A_inv.dot(B))
        x_w = np.linalg.solve(S, dw)
        x_u = np.linalg.solve(S, B.dot(A_inv.dot(du)))
        direction = np.vstack((x_w - x_u, -A_inv.dot(B.dot(x_w)) + A_inv.dot(du + B.dot(x_u))))
        return -direction

    n, d = X.shape
    w = np.zeros((d, 1))
    if not w_list is None:
        w_list.append(np.copy(w))
    if not time_list is None:
        time_list.append(0)
    u = np.ones((d, 1))
    point = np.vstack((w, u))
    seq = tau_sequence()
    if display:
        'Primal Barrier Method'
    start_time = time.time()
    for outer_iter in range(max_iter):
        tau = next(seq)
        for inner_iter in range(max_inner_iter):
            loss, grad, hess = oracle(point, tau)
            if display:
                print('\tInner Iteraion', inner_iter, ': gradient norm = ', np.linalg.norm(grad))
            if np.linalg.norm(grad) < tol_center:
                break
            direction = newton_direction(hess, grad)
            # step = armiho(lambda w: oracle(w, tau), point, direction, bt_sequence(), a_eps=bt_params[0])            # step = armiho(lambda w: oracle(w, tau), point, direction, bt_sequence(), a_eps=bt_params[0])
            step = step_size(point, direction)
            point += step * direction

        w = point[:d, :]
        if not time_list is None:
            time_list.append(time.time() - start_time)
        if not w_list is None:
            w_list.append(np.copy(w))
        mu = X.dot(w) - t
        mu = reg_coef * mu / np.linalg.norm(X.T.dot(mu), ord=np.inf)
        gap = np.linalg.norm(t - X.dot(w))**2 / 2 + reg_coef * np.linalg.norm(w, 1) + np.linalg.norm(mu)**2 / 2 + \
              mu.T.dot(t)
        if display:
            print('Outer iteration', outer_iter, ': gap =', gap)
        if gap < tol_gap:
            return w
    return w

def pd(X, t, reg_coef, max_iter=100, tol_feas=1e-10, tol_gap=1e-5, tau_param=10,
bt_params=np.array([1e-4, 0.8]), display=0, mu_list=None, time_list=None):
    """
    Primal-Dual method
    :param X: (NxD) numpy.array, training set
    :param t: (N,) numpy.array, target values
    :param reg_coef: regularization coefficient
    :param max_iter: maximum iteration number
    :param tol_feas:
    :param tol_gap: tolerance for gap between the values of the primal and dual problem
    :param tau_param: numpy.array([tau_start, nu]), strategy for centring parameter tau_n = tau_start * nu
    :param bt_params: numpy.array([c_1, beta]), backtracking strategy parameters
    :param display: a number, showing weather or not to display progress iteration-wise
    :param w_list: a list for saving the method's progress
    :param time_list: a list for saving the iteration-wise times
    :return w: (D, 1) numpy.array, optimal weights
    :return mu: (D, 1) numpy.array, optimal dual variables
    """
    if not isinstance(X, np.ndarray) or not isinstance(t, np.ndarray) or not isinstance(bt_params, np.ndarray):
        raise TypeError("X, t, bt_params and tau_params must be numpy arrays")
    if X.shape[0] != t.shape[0]:
        raise ValueError("X.shape[0] != t.shape[0]")
    if len(t.shape) == 1:
        t = t[:, None]
    elif t.shape[1] != 1.:
        raise ValueError("t.shape must be of the form (N,) or (N, 1)")

    def residual(point):
        """
        The residual vector
        :param point: variable vector (mu, gamma_1, gamma_2)
        :return: r, residual
        """
        w = point[:n, :]
        lambda_1 = point[n:n+d, :]
        lambda_2 = point[n+d:, :]
        if np.any(lambda_1 < 0) or np.any(lambda_2 < 0):
            return None
        return -np.vstack((w + t + X.dot(lambda_1 - lambda_2), lambda_1 * (X.T.dot(w) - reg_coef) + 1 / tau,
                           lambda_2 * (-X.T.dot(w) - reg_coef) + 1 / tau))

    def tau_sequence():
        """
        Generator for the sequence of centering parameters tau
        :return: tau
        """
        res = 1.
        while True:
            yield res
            res *= tau_param

    def matrix(point):
        """
        Linearized KKT system matrix
        :param point: variable vector (mu, gamma_1, gamma_2)
        :return:
        """
        w = point[:n, :]
        lambda_1 = point[n:n+d, :]
        lambda_2 = point[n+d:, :]
        mat = np.hstack((np.eye(n), X, -X))
        mat = np.vstack((mat, np.hstack((lambda_1 * X.T, np.diag((X.T.dot(w) - reg_coef)[:, 0]),
                                         np.zeros((d, d))))))
        mat = np.vstack((mat, np.hstack((-lambda_2 * X.T, np.zeros((d, d)),
                                         np.diag((-X.T.dot(w) - reg_coef)[:, 0])))))
        return mat


    def direction(mat, res):
        """
        Optimization direction
        :param mat: left-hand side of the linearised KKt
        :param res: right-hand side of the linearised KKT
        :return: optimization direction
        """
        C = np.hstack((X, -X))
        D = mat[n:, :n]
        E = mat[n:, n:]
        S = E - D.dot(C)
        r_d = res[:n, :]
        r_c = res[n:, :]
        x_d = np.linalg.solve(S, D.dot(r_d))
        x_c = np.linalg.solve(S, r_c)
        return np.vstack((r_d + C.dot(x_d) - C.dot(x_c), x_c - x_d))

    def step(point, direction):
        step = 1.
        eps = bt_params[0]
        current_loss = np.linalg.norm(residual(point))
        new_point = point + step * direction
        new_loss = residual(new_point)
        while new_loss is None:
                step *= bt_params[1]
                new_point = point + step * direction
                new_loss = residual(new_point)
        while np.linalg.norm(new_loss) > current_loss * (1 - step * eps):
            step *= bt_params[1]
            new_point = point + step * direction
            new_loss = residual(new_point)
            while new_loss is None:
                step *= bt_params[1]
                new_point = point + step * direction
                new_loss = residual(new_point)
        return step

    n, d = X.shape
    mu = np.zeros((n, 1))*0.1
    gamma_1 = np.ones((d, 1))*0.7
    gamma_2 = np.ones((d, 1))*0.7
    point = np.vstack((mu, gamma_1, gamma_2))
    seq = tau_sequence()
    if display:
        'Primal-Dual Method'
    if not mu_list is None:
        mu_list.append(mu)
    if not time_list is None:
        time_list.append(0)
    start_time = time.time()
    for iteration in range(max_iter):
        tau = next(seq)
        mat = matrix(point)
        res = residual(point)
        r_d = res[:n, :]
        if display:
            print('Iteration', iteration, ': ')
            print('\tDual residual norm:', np.linalg.norm(r_d))
        if np.linalg.norm(r_d) < tol_feas:
            mu = point[:n, :]
            gamma_1 = point[n:n+d, :]
            gamma_2 = point[n+d:, :]
            margin = gamma_1.T.dot(X.T.dot(mu) - reg_coef) - gamma_2.T.dot(X.T.dot(mu) + reg_coef)
            if display:
                print('\tSurrogate margin:', margin)
            if margin < tol_gap:
                w = np.linalg.lstsq(X, mu + t)[0]
                return w, mu
        direct = direction(mat, res)
        alpha = step(point, direct)
        point += alpha * direct
        if not time_list is None:
            time_list.append(time.time() - start_time)
        if not mu_list is None:
            mu_list.append(np.copy(mu))
    mu = point[:n, :]
    w = np.linalg.lstsq(X, mu + t)[0]
    return w, mu

def prox(X, t, reg_coef, max_iter=100, tol_gap=1e-5, display=0, w_list=None, time_list=None):
    """
    Primal barrier method
    :param X: (NxD) numpy.array, training set
    :param t: (N,) numpy.array, target values
    :param reg_coef: regularization coefficient
    :param max_iter: maximum iteration number
    :param tol_gap: tolerance for gap between the values of the primal and dual problem
    :param display: a number, showing weather or not to display progress iteration-wise
    :param w_list: a list for saving the method's progress
    :param time_list: a list for saving the iteration-wise times
    :return w: (D,) numpy.array, optimal weights
    """

    if not isinstance(X, np.ndarray) or not isinstance(t, np.ndarray):
        raise TypeError("X, t must be numpy arrays")
    if X.shape[0] != t.shape[0]:
        raise ValueError("X.shape[0] != t.shape[0]")
    if len(t.shape) == 1:
        t = t[:, None]
    elif t.shape[1] != 1.:
        raise ValueError("t.shape must be of the form (N,) or (N, 1)")

    n, d = X.shape
    w = np.zeros((d, 1))
    if not w_list is None:
            w_list.append(np.copy(w))
    if not time_list is None:
        time_list.append(0)
    l = 30000.
    anc_mat = 2 * X.T.dot(t)
    anc_mat_2 = 2 * X.T.dot(X)
    if display:
        print('Proximal Method')
    start_time = time.time()
    for iteration in range(max_iter):
        # l *= 10
        b = anc_mat_2.dot(w) - l * w - anc_mat
        w = np.zeros(b.shape)

        indices = np.abs(b) > reg_coef
        b_indices = b[indices]
        w[indices] = (-b_indices + np.sign(b_indices) * reg_coef) / l
        mu = X.dot(w) - t
        mu = reg_coef * mu / np.linalg.norm(X.T.dot(mu), ord=np.inf)
        gap = np.linalg.norm(t - X.dot(w))**2 / 2 + reg_coef * np.linalg.norm(w, 1) + np.linalg.norm(mu)**2 / 2 + \
              mu.T.dot(t)
        if not w_list is None:
            w_list.append(np.copy(w))
        if not time_list is None:
            time_list.append(time.time() - start_time)
        if display:
            print('Iteration', iteration, ': gap =', gap)
        if gap < tol_gap:
            return w
    return w

