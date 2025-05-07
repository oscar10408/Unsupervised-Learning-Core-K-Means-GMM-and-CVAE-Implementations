"""EECS545 HW5 Q1. K-means"""

import numpy as np
from typing import NamedTuple, Union, Literal
from scipy.stats import multivariate_normal
from scipy.special import logsumexp

def hello():
    print('Hello from gmm.py!')


class GMMState(NamedTuple):
    """Parameters to a GMM Model."""
    pi: np.ndarray  # [K]
    mu: np.ndarray  # [K, d]
    sigma: np.ndarray  # [K, d, d]


def train_gmm(train_data: np.ndarray,
              init_pi: np.ndarray,
              init_mu: np.ndarray,
              init_sigma: np.ndarray,
              *,
              num_iterations: int = 50,
              ) -> GMMState:
    """Fit a GMM model.

    Arguments:
        train_data: A numpy array of shape (N, d), where
            N is the number of data points
            d is the dimension of each data point. Note: you should NOT assume
              d is always 3; rather, try to implement a general K-means.
        init_pi: The initial value of pi. Shape (K, )
        init_mu: The initial value of mu. Shape (K, d)
        init_sigma: The initial value of sigma. Shape (K, d, d)
        num_iterations: Run EM (E-steps and M-steps) for this number of
            iterations.

    Returns:
        A GMM parameter after running `num_iterations` number of EM steps.
    """
    # Sanity check
    N, d = train_data.shape
    K, = init_pi.shape
    assert init_mu.shape == (K, d)
    assert init_sigma.shape == (K, d, d)


    
    pi, mu, sigma = init_pi.copy(), init_mu.copy(), init_sigma.copy()
    ###########################################################################
    # Implement EM algorithm for learning GMM.
    ###########################################################################
    
    # TODO: Add your implementation.
    # Feel free to add helper functions as much as needed.
    
    for i in range(num_iterations):
        gamma, log_totals = compute_gamma(pi, mu, sigma, train_data) 
        num_points_in_clusters = gamma.sum(axis=1, keepdims=True)
        pi = (num_points_in_clusters / N).reshape([K])
        mu = np.matmul(gamma, train_data) / num_points_in_clusters

        gamma_x_sum = np.matmul(gamma, train_data)
        mu = gamma_x_sum / num_points_in_clusters

        log_likelihood = np.sum(log_totals)
        print(f'Iteration {i:2d}: log-likelihood={log_likelihood:6.2f}')

        for k in range(K):
            diff = train_data - mu[k]
            sigma[k] = (gamma[k, :, None] * diff).T @ diff / num_points_in_clusters[k]

    #######################################################################

    return GMMState(pi, mu, sigma)

def compute_gamma(pi, mu, sigma, data):
    K, num_data = mu.shape[0], data.shape[0]
    log_normpdf = np.array([
        multivariate_normal(mu[k], sigma[k]).logpdf(data) for k in range(K)
    ])
    
    log_pi_normpdf = np.log(pi + 1e-10).reshape(-1, 1) + log_normpdf
    log_totals = logsumexp(log_pi_normpdf, axis=0, keepdims=True)
    
    gamma = np.exp(log_pi_normpdf - log_totals)
    return gamma, log_totals


def compress_image(image: np.ndarray, gmm_model: GMMState) -> np.ndarray:
    """Compress image by mapping each pixel to the mean value of a
    Gaussian component (hard assignment).

    Arguments:
        image: A numpy array of shape (H, W, 3) and dtype uint8.
        gmm_model: type GMMState. A GMM model parameters.
    Returns:
        compressed_image: A numpy array of (H, W, 3) and dtype uint8.
            Be sure to round off to the nearest integer.
    """
    H, W, C = image.shape
    K = gmm_model.mu.shape[0]
    ###########################################################################
    # Implement image compression algorithm using the GMM model
    ##########################################################################
    # TODO: Add your implementation.
    # Feel free to add helper functions as much as needed.

    test_data = image.reshape(-1, C)
    pi, mu, sigma = gmm_model

    gamma, _ = compute_gamma(pi, mu, sigma, test_data)
    group = np.argmax(gamma, axis=0)    

    compressed = test_data.copy()
    for idx in range(K):
        mask = (group == idx)
        compressed[mask, :] = mu[idx]

    compressed_image = compressed.reshape([H, W, C])

    ##########################################################################

    assert compressed_image.dtype == np.uint8
    assert compressed_image.shape == (H, W, C)
    return compressed_image
