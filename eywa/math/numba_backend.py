from numba import jit, prange
import numpy as np
import numba
import os
import sys
from time import time


num_cores = numba.config.NUMBA_DEFAULT_NUM_THREADS


py3 = sys.version_info[0] == 3

parallel = True
if os.name == 'nt' and not py3:
    parallel=False

@jit(nopython=True, fastmath=True, parallel=parallel)
def _soft_identity_matrix(matrix, nx, ny):
    for i in prange(nx):
        for j in prange(ny):
            matrix[i, j] = 1. / (np.abs(i - j) + 1.)


@jit(nopython=False, fastmath=True, parallel=parallel)
def  soft_identity_matrix(nx, ny):
    m = np.empty((nx, ny), dtype='float32')
    _soft_identity_matrix(m, nx, ny)
    return m


@jit(nopython=True, fastmath=True, parallel=parallel)
def __vector_sequence_similarity_euclid(x, y, z, nx, ny, locality=0.5):
    nx = len(x)
    ny = len(y)
    #z = np.array([[0. for __ in range(ny)] for _ in range(nx)])
    m1 = 0.
    m2 = 0.
    for i in prange(nx):
        xi = x[i]
        for j in prange(ny):
            yj = y[j]
            zij = (1. - (((xi - yj) ** 2).sum() ** 0.5))
            seye = 1. / (np.abs(i - j) + 1)
            zij *= locality * (seye - 1) + 1
            z[i, j] = zij
        m2 += z[i, :].max()
    for j in prange(ny):
        m1 += z[:, j].max()
    return (m1 + m2) / (nx + ny)


@jit
def _vector_sequence_similarity_euclid(x, y, locality=0.5):
    nx = len(x)
    ny = len(y)
    z = np.empty((nx, ny), dtype='float32')
    return __vector_sequence_similarity_euclid(x, y, z, nx, ny, locality=locality)


@jit(nopython=True, fastmath=True, parallel=parallel)
def _vector_sequence_similarity_dot(x, y, locality=0.5):
    nx = len(x)
    ny = len(y)
    z = np.dot(x, y.T)
    m2 = 0.
    for i in prange(nx):
        for j in prange(ny):
            seye = 1. / (np.abs(i - j) + 1)
            zij = z[i, j] * (locality * (seye - 1) + 1)
            z[i, j] = zij
        m2 += z[i, :].max()
    m1 = 0.
    for j in prange(ny):
        m1 += z[:, j].max()
    return 0.5 * (m1 + m2) / (nx + ny)

@jit
def vector_sequence_similarity(x, y, locality=0.5, metric='dot'):
    assert metric in ('dot', 'euclid')
    if metric == 'dot':
        return _vector_sequence_similarity_dot(x, y, locality)
    elif metric == 'euclid':
        return _vector_sequence_similarity_euclid(x, y, locality)


@jit(nopython=True, fastmath=True, parallel=parallel)
def _batch_vector_sequence_similarity(X, y):
    batch_size = len(X)
    output = []
    mini_batch_size = num_cores
    done = 0
    rem = batch_size % mini_batch_size
    batch_size -= rem
    ny = len(y)
    locality = 0.5 # hard
    while done < batch_size:
        for idx in prange(done, done + mini_batch_size):
            x = X[idx]
            nx = len(x)
            z = np.dot(x, y.T)
            m2 = 0.
            for i in prange(nx):
                for j in prange(ny):
                    seye = 1. / (np.abs(i - j) + 1)
                    zij = z[i, j] * (locality * (seye - 1) + 1)
                    z[i, j] = zij
                m2 += z[i, :].max()
            m1 = 0.
            for j in prange(ny):
                m1 += z[:, j].max()
            output.append(0.5 * (m1 + m2) / (nx + ny))
        done += mini_batch_size
    for idx in range(done, done + rem):
            x = X[idx]
            nx = len(x)
            z = np.dot(x, y.T)
            m2 = 0.
            for i in prange(nx):
                for j in prange(ny):
                    seye = 1. / (np.abs(i - j) + 1)
                    zij = z[i, j] * (locality * (seye - 1) + 1)
                    z[i, j] = zij
                m2 += z[i, :].max()
            m1 = 0.
            for j in prange(ny):
                m1 += z[:, j].max()
            output.append(0.5 * (m1 + m2) / (nx + ny))
    return output

def batch_vector_sequence_similarity(X, y):
    # TODO: vectorize
    if len(y) == 0:
        return [int(len(x) == 0) for x in X]
    return _batch_vector_sequence_similarity(X, y)

@jit
def euclid_distance(x, y):
    return ((x - y) ** 2).sum() ** 0.5

@jit
def euclid_similarity(x, y):
    return 1. - ((x - y) ** 2).sum() ** 0.5

@jit
def softmax(x, axis=None):
    e = np.exp(x - x.max())
    s = e.sum(axis=axis, keepdims=True)
    e /= s
    return e

@jit
def frequencies_to_weights(x):
    return softmax(1. - softmax(x))
