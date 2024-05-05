"""
A module of utility functions.
"""
import torch
import math
import numpy as np

def normal_logpdf(x, mu, sigmasq):
    """
    Evaluate log-density function of a normal distribution.

    Parameters
    ----------   
    x : evaluation states (of size d or N x d)

    mu : mean vector (of size 1 x d or N x d)

    sigmasq : scalar variance 
                        
    Returns
    -------    
    logdensity : log-density values(N, 1)
    """
    
    d = mu.shape[1]
    constants = - 0.5 * d * torch.log(torch.tensor(2 * math.pi, device = x.device)) - 0.5 * d * torch.log(sigmasq)
    logdensity = torch.squeeze(constants - 0.5 * torch.sum((x - mu)**2, 1) / sigmasq)
    
    return logdensity

def inverse_cdf(su, W):
    """Inverse CDF algorithm for a finite distribution. 
    (From particles package of Nicolas Chopin)
        
    Parameters
    ----------
    su: (M,) ndarray
        M sorted uniform variates (i.e. M ordered points in [0,1]).
    W: (N,) ndarray
        a vector of N normalized weights (>=0 and sum to one)
    Returns
    -------
    A: (M,) ndarray
        a vector of M indices in range 0, ..., N-1
    """
    j = 0
    s = W[0]
    M = su.shape[0]
    A = np.empty(M, dtype=np.int64)
    for n in range(M):
        while su[n] > s:
            j += 1
            s += W[j]
        A[n] = j
    return A


def uniform_spacings(N):
    """ Generate ordered uniform variates in O(N) time.
    (From particles package of Nicolas Chopin)
    
    Parameters
    ----------
    N: int (>0)
        the expected number of uniform variates
    Returns
    -------
    (N,) float ndarray
        the N ordered variates (ascending order)
    Note
    ----
    This is equivalent to::
        from numpy import random
        u = sort(random.rand(N))
    but the line above has complexity O(N*log(N)), whereas the algorithm
    used here has complexity O(N).
    """
    z = np.cumsum(-np.log(np.random.rand(N + 1)))
    return z[:-1] / z[-1]


def resampling(W, M):
    """
    Multinomial resampling scheme. 
    (From particles package of Nicolas Chopin)
    
    Parameters
    ----------    
    W: (N,) ndarray
        a vector of N normalized weights (>=0 and sum to one)
    
    M: int
        number of ancestor indexes to be sampled
    
    Returns
    -------
    A: (N,) ndarray
        a vector of N indices in range 0, ..., N-1
    """  
    
    return torch.from_numpy(inverse_cdf(uniform_spacings(M), W))

