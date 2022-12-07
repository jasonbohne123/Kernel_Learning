import numpy as np
from sklearn import svm


def compose_kernels(X, kernel_list, weights):
    """ Compute positive linear combination of kernels 
    """
    return np.sum(np.array([weights[ct]*k_i.compute_kernel(X) for ct, k_i in enumerate(kernel_list)]), axis=0)


def compute_dual(X, y, kernel_list, d_m, constant, compute_gap=True):
    """ Compute dual objective value for single SVM
    """
    kernel = compose_kernels(X, kernel_list, d_m)
    single_kernel = svm.SVC(C=constant, kernel='precomputed')
    single_kernel.fit(kernel, y)

    alpha = np.empty(len(y))
    alpha[single_kernel.support_] = np.abs(single_kernel.dual_coef_[0])
    alpha[alpha == None] = 0

    y_outer = np.outer(y, y)

    J = -0.5*np.dot(np.dot(alpha, np.multiply(y_outer, kernel)),
                    alpha.T)+np.sum(alpha)

    if compute_gap:
        kernel_eval = [np.dot(np.dot(alpha, np.multiply(
            y_outer, k_i.compute_kernel(X))), alpha.T) for k_i in kernel_list]
        duality_gap = J-np.sum(alpha)+0.5*np.max(kernel_eval)

        return J, duality_gap, alpha

    return J


def compute_gradient(kernel, X, y_outer, alpha):
    """ Compute gradient of MKL objective closed form ; vector
    """
    kernel_mat = kernel.compute_kernel(X)
    gradient_obj = -0.5 * \
        np.dot(np.dot(alpha, np.multiply(y_outer, kernel_mat)), alpha.T)

    return gradient_obj


def descent_direction(d_m, mu, gradient_j, grad_mu):
    """ Compute direction of gradient descent ; vector 
    """
    n = len(d_m)
    D = np.zeros(n)
    ongoing_sum = 0
    for index in range(0, n):
        if d_m[index] == 0 and gradient_j[index]-grad_mu > 0:
            D[index] = 0

        elif d_m[index] > 0 and index != mu:

            grad_m = -gradient_j[index]+grad_mu
            D[index] = grad_m
            ongoing_sum += grad_m

        else:
            D[index] = 0

    D[mu] = -ongoing_sum

    return D


def line_search(X, y, kernel_list, D, d_m, gamma_max, disc):
    """ Selects step size to minimize obj value;  

        Update from heuristic to exact Armijo's rule 
    """

    if gamma_max == 0:
        return gamma_max

    # grid of step size begins bigger than 0
    grid = np.arange(0+gamma_max/disc, gamma_max, gamma_max/disc)

    min_gamma, min_obj_val = None, 10e8
    for gamma_i in grid:
        d_i = d_m+gamma_i*D
        dual_obj_val = compute_dual(
            X, y, kernel_list, d_i, constant=100, compute_gap=False)

        if abs(dual_obj_val) < abs(min_obj_val):
            min_obj_val = dual_obj_val
            min_gamma = gamma_i

    return min_gamma


def form_optimal_svm(kernel_list, d_m, X, y):
    """ Forms optimal SVM from optimal weights and kernels
    """
    # compute optimal kernel
    K = np.zeros((X.shape[0], X.shape[0]))
    for i in range(len(kernel_list)):
        K += d_m[i]*kernel_list[i].compute_kernel(X)

    # fit svm
    clf = svm.SVC(kernel='precomputed')
    clf.fit(K, y)
    return clf, K
