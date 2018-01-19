import numpy as np


def soft_identity_matrix(nx, ny):
     return 1. / np.array([[abs(i - j) + 1 for j in range(ny)] for i in range(nx)])


def vector_sequence_similarity(x, y):
    nx = len(x)
    ny = len(y)
    z = x.dot(y.T)
    mag = ((x ** 2).sum(1, keepdims=True) *
          (y ** 2).sum(1, keepdims=True).T) ** 0.5
    z /= mag
    z *= soft_identity_matrix(nx, ny)
    m1 = z.max(axis=0).sum()
    m2 = z.max(axis=1).sum()
    return 0.5 * (m1 + m2) / (nx + ny)
