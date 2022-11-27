from scipy.spatial import distance_matrix
import numpy as np


class Kernel():
    """ Abstract method to compute the kernel matrix under gaussian kernel or polynomial kernel"""

    def __init__(self, kernel_type, order=None, bandwidth=None):
        self.kernel_type = kernel_type
        self.order = order
        self.bandwidth = bandwidth

    def compute_kernel(self, X):

        if self.kernel_type == 'linear':
            return np.dot(X, X.T)

        if self.kernel_type == 'gaussian':
            # Compute the distance matrix which is then numerically transformed to gaussian kernel
            scale = distance_matrix(X, X, p=2)
            return np.exp(-0.5*self.bandwidth*(scale**2))

        if self.kernel_type == 'polynomial':
            return np.dot(X, X.T)**self.order
