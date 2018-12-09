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
    parallel = False


@jit('(f8[:, :], i4, i4)', nopython=True, fastmath=True, parallel=parallel)
def _soft_identity_matrix(matrix, nx, ny):
    for i in prange(nx):
        for j in prange(ny):
            matrix[i, j] = 1. / (np.abs(i - j) + 1.)


@jit(nopython=False, fastmath=True, parallel=parallel)
def soft_identity_matrix(nx, ny):
    m = np.empty((nx, ny), np.float32)
    _soft_identity_matrix(m, nx, ny)
    return m


@jit(nopython=True, fastmath=True, parallel=parallel)
def __vector_sequence_similarity_euclid(x, y, z, nx, ny, locality=0.5):
    nx = len(x)
    ny = len(y)
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
    z = np.empty((nx, ny), np.float32)
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


def vector_sequence_similarity(x, y, locality=0.5, metric='dot'):
    assert metric in ('dot', 'euclid')
    if metric is 'dot':
        return _vector_sequence_similarity_dot(x, y, locality)
    elif metric is 'euclid':
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
    locality = 0.5  # hard
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
            output.append(float(0.5 * (m1 + m2) / (nx + ny)))
        done += mini_batch_size
    if rem > 0:
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
            output.append(float(0.5 * (m1 + m2) / (nx + ny)))
    return output


def batch_vector_sequence_similarity(X, y):
    # TODO: vectorize
    if len(y) == 0:
        return [int(len(x) == 0) for x in X]
    return _batch_vector_sequence_similarity(X, y)


@jit
def euclid_distance(x, y):
    return (np.subtract(x, y) ** 2).sum() ** 0.5


@jit
def euclid_similarity(x, y):
    return np.subtract(1., (np.subtract(x, y) ** 2).sum() ** 0.5)


@jit(nopython=True, fastmath=True, parallel=parallel)
def _softmax2d(x, axis=-1):
    assert x.ndim == 2
    shape = x.shape
    if axis < 0:
        axis += 2
    assert axis < 2
    x = x.copy()
    tensors = []
    if axis == 0:
        xt = x.T
        for i in range(shape[1]):
            tensors.append(xt[i])
        for t in tensors:
            t -= t.max()
            np.exp(t, t)
            t /= t.sum()
    else:
        for i in range(shape[0]):
            tensors.append(x[i])
        for t in tensors:
            t -= t.max()
            np.exp(t, t)
            t /= t.sum()
    return x


@jit(nopython=True, fastmath=True, parallel=parallel)
def _softmax1d(x):
    assert x.ndim == 1
    x = x.copy()
    x -= x.max()
    np.exp(x, x)
    x /= x.sum()
    return x


@jit(nopython=False, forceobj=True)
def softmax(x, axis=-1):
    ndim = x.ndim
    if ndim == 1:
        return _softmax1d(x)
    elif ndim == 2:
        return _softmax2d(x, axis)
    else:
        raise NotImplementedError("In numba backend, softmax supports only 1d and 2d inputs.")


@jit
def frequencies_to_weights(x):
    return softmax(1. - softmax(x))


# entity extractor

@jit(nopython=False, forceobj=True)
def should_pick(x_embs, pick_embs, non_pick_embs, variance, weights):
    npicks = len(pick_embs)
    scores = _batch_vector_sequence_similarity(pick_embs + non_pick_embs, x_embs)
    pick_score = scores[0]
    for i in range(1, npicks):
        s = scores[i]
        if s > pick_score:
            pick_score = s
    if non_pick_embs:
        non_pick_score = scores[npicks]
        for i in range(npicks + 1, len(scores)):
            s = scores[i]
            if s > non_pick_score:
                non_pick_score = s
    else:
        non_pick_score = scores[0] * 0.
    pick_bias = weights[3]
    s = pick_score + non_pick_score
    pick_score /= s
    non_pick_score /= s
    pick_score *= pick_bias
    pick_score *= variance
    non_pick_score *= 1. - pick_bias
    non_pick_score += 1. - variance
    return pick_score >= non_pick_score


@jit(nopython=True, fastmath=True, parallel=parallel)
def get_token_score(token_emb, token_left_embs, token_right_embs, lefts_embs, rights_embs, vals_embs, is_entity,
                    weights):
    if len(token_left_embs) == 0:
        left_scores = []
        for i in prange(len(lefts_embs)):
            x = lefts_embs[i]
            if len(x) == 0:
                left_scores.append(1.)
            else:
                left_scores.append(0.)
    else:
        left_scores = _batch_vector_sequence_similarity(lefts_embs, token_left_embs)
    left_score = left_scores[0]
    for i in range(1, len(left_scores)):
        s = left_scores[i]
        if s > left_score:
            left_score = s

    if len(token_right_embs) == 0:
        right_scores = []
        for i in prange(len(rights_embs)):
            x = rights_embs[i]
            if len(x) == 0:
                right_scores.append(1.)
            else:
                right_scores.append(0.)
    else:
        right_scores = _batch_vector_sequence_similarity(rights_embs, token_right_embs)

    right_score = right_scores[0]
    for i in range(1, len(right_scores)):
        s = right_scores[i]
        if s > right_score:
            right_score = s

    value_scores = []
    for i in range(len(vals_embs)):
        val_emb = vals_embs[i]
        value_scores.append(np.subtract(1., (np.subtract(val_emb, token_emb) ** 2).sum() ** 0.5))
    value_score = value_scores[0]
    for i in range(1, len(value_scores)):
        s = value_scores[i]
        if s > value_score:
            value_score = s

    left_right_weight = weights[0]
    word_neighbor_weight = weights[1]
    neighbor_score = left_right_weight * left_score + (1. - left_right_weight) * right_score
    token_score = word_neighbor_weight * value_score + (1. - word_neighbor_weight) * neighbor_score
    if is_entity:
        token_score *= 1. + weights[2]
    return token_score


# TODO:fix this
if 'TRAVIS' in os.environ:
    from .numpy_backend import get_token_score
