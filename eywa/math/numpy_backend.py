from numpy import *
import numpy as np

py_max = max
max = np.max
py_sum = sum
sum = np.sum


def soft_identity_matrix(nx, ny):
    return 1. / array([[abs(i - j) + 1 for j in range(ny)] for i in range(nx)])


def vector_sequence_similarity(x, y, locality=0.5, metric='dot'):
    assert metric in ('dot', 'euclid')
    if metric == 'dot':
        return _vector_sequence_similarity_dot(x, y, locality)
    elif metric == 'euclid':
        return _vector_sequence_similarity_euclid(x, y, locality)


def _vector_sequence_similarity_euclid(x, y, locality=0.5):
    nx = len(x)
    ny = len(y)
    x = expand_dims(x, 1)
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
    return 1. - sum((subtract(x, y) ** 2), -1) ** 0.5


def softmax(x, axis=-1):
    x = x.copy()
    mx = max(x, axis, keepdims=True)
    x -= mx
    exp(x, x)
    s = sum(x, axis, keepdims=True)
    x /= s
    return x


def frequencies_to_weights(x):
    return softmax(1. - softmax(x))


# entity extractor

def should_pick(x_embs, pick_embs, non_pick_embs, variance, weights):
    npicks = len(pick_embs)
    scores = batch_vector_sequence_similarity(pick_embs + non_pick_embs, x_embs)
    pick_score = max(scores[:npicks])
    if non_pick_embs:
        non_pick_score = max(scores[npicks:])
    else:
        non_pick_score = 0.
    pick_bias = weights[3]
    s = pick_score + non_pick_score
    pick_score /= s
    non_pick_score /= s
    pick_score *= pick_bias
    pick_score *= variance
    non_pick_score *= 1. - pick_bias
    non_pick_score += 1. - variance
    return pick_score >= non_pick_score


def get_token_score(token_emb, token_left_embs, token_right_embs, lefts_embs, rights_embs, vals_embs, is_entity,
                    weights):
    left_score = max(batch_vector_sequence_similarity(lefts_embs, token_left_embs))
    right_score = max(batch_vector_sequence_similarity(rights_embs, token_right_embs))
    value_score = max(euclid_similarity(vals_embs, token_emb))
    left_right_weight = weights[0]
    word_neighbor_weight = weights[1]
    neighbor_score = left_right_weight * left_score + (1. - left_right_weight) * right_score
    token_score = word_neighbor_weight * value_score + (1. - word_neighbor_weight) * neighbor_score
    if is_entity:
        token_score *= 1. + weights[2]
    return token_score
