import numpy as np


def soft_identity_matrix(nx, ny):
     return 1. / np.array([[abs(i - j) + 1 for j in range(ny)] for i in range(nx)])


def vector_sequence_similarity(x, y, locality=0.5, metric='dot'):
    assert metric in ('dot', 'euclid')
    if metric == 'dot':
        return _vector_sequence_similarity_dot(x, y, locality)
    elif metric == 'euclid':
        return _vector_sequence_similarity_euclid(x, y, locality)


def _vector_sequence_similarity_euclid(x, y, locality=0.5):
    nx = len(x)
    ny = len(y)
    x = np.expand_dims(x, 1)
    z = 1. - ((x - y) ** 2).sum(2) ** 0.5
    z *= locality * (soft_identity_matrix(nx, ny) - 1) + 1.
    m1 = z.max(axis=0).sum()
    m2 = z.max(axis=1).sum()
    return (m1 + m2) / (nx + ny)  

def _vector_sequence_similarity_dot(x, y, locality=0.5):
    nx = len(x)
    ny = len(y)
    z = x.dot(y.T)
    '''
    mag = ((x ** 2).sum(1, keepdims=True) *
          (y ** 2).sum(1, keepdims=True).T) ** 0.5
    z /= mag
    '''
    z *= locality * (soft_identity_matrix(nx, ny) - 1) + 1.
    m1 = z.max(axis=0).sum()
    m2 = z.max(axis=1).sum()
    return 0.5 * (m1 + m2) / (nx + ny)


def batch_vector_sequence_similarity(X, y):
    # TODO: vectorize
    if len(y) == 0:
        return [int(len(x) == 0) for x in X]
    return [0 if len(x) == 0 else vector_sequence_similarity(x, y) for x in X]


def euclid_distance(x, y):
    return ((x - y) ** 2).sum() ** 0.5

def euclid_similarity(x, y):
    return 1. - ((x - y) ** 2).sum() ** 0.5


def softmax(x, axis=None):
    e = np.exp(x - x.max())
    s = e.sum(axis=axis, keepdims=True)
    e /= s
    return e


def frequencies_to_weights(x):
    return softmax(1. - softmax(x))
