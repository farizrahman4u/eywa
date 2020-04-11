import tensorflow as tf
import numpy as np


K = tf.keras.backend


def soft_identity_matrix(nx, ny):
    m = tf.range(nx * ny)
    m = tf.cast(m, 'float32')
    r = m // ny
    c = m  % nx
    return tf.reshape((1./ (1 + K.abs(r - c))), (nx, ny))


def vector_sequence_similarity(x, y, locality=0.5, metric='dot'):
    assert metric in ('dot', 'euclid')
    if metric == 'dot':
        return _vector_sequence_similarity_dot(x, y, locality)
    elif metric == 'euclid':
        return _vector_sequence_similarity_euclid(x, y, locality)


def _vector_sequence_similarity_euclid(x, y, locality=0.5):
    nx = len(x)
    ny = len(y)
    x = tf.expand_dims(x, 1)
    z = 1. - tf.pow(tf.reduce_sum(tf.pow((x - y), 2), 2), 0.5)
    m1 = tf.reduce_sum(tf.reduce_max(z, axis=0))
    m2 = tf.reduce_sum(tf.reduce_max(z, axis=1))
    return ((m1 + m2) / (nx + ny))


def _vector_sequence_similarity_dot(x, y, locality=0.5):
    nx = len(x)
    ny = len(y)
    z = tf.matmul(x, y, False, True)
    '''
    mag = ((x ** 2).sum(1, keepdims=True) *
          (y ** 2).sum(1, keepdims=True).T) ** 0.5
    z /= mag
    '''
    z = z * locality * (soft_identity_matrix(nx, ny) - 1) + 1.
    m1 = tf.reduce_sum(tf.reduce_max(z, axis=0))
    m2 = tf.reduce_sum(tf.reduce_max(z, axis=1))
    return ((m1 + m2) / (nx + ny))


def batch_vector_sequence_similarity(X, y):
    z = []
    for x in X:
        z.append(_vector_sequence_similarity_dot(x, y))
    return tf.stack(z)



def euclid_distance(x, y):
    return tf.pow(tf.reduce_sum(tf.pow(x - y, 2)), 0.5)


def euclid_similarity(x, y):
    return (1. - tf.pow(tf.reduce_sum(tf.pow(x - y, 2)), 0.5))


def softmax(x, axis=-1):
    return K.softmax(x, axis)


def frequencies_to_weights(x):
    return softmax(1. - softmax(x))


# entity extractor

def should_pick(x_embs, pick_embs, non_pick_embs, variance, weights):
    # returns positive if should pick
    npicks = len(pick_embs)
    scores = batch_vector_sequence_similarity(pick_embs + non_pick_embs, x_embs)
    pick_score = tf.reduce_max(scores[:npicks])
    if non_pick_embs:
        non_pick_score = tf.reduce_max(scores[npicks:])
    else:
        non_pick_score = 0.
    pick_bias = weights[3]
    s = pick_score + non_pick_score
    pick_score /= s
    non_pick_score /= s
    pick_score *= pick_bias
    pick_score *= variance
    non_pick_score *= 1. - pick_bias
    non_pick_score *= 1. - variance
    return pick_score - non_pick_score


def get_token_score(token_emb, token_left_embs, token_right_embs, lefts_embs, rights_embs, vals_embs, is_entity,
                    weights):
    left_score = tf.reduce_max(batch_vector_sequence_similarity(lefts_embs, token_left_embs))
    right_score = tf.reduce_max(batch_vector_sequence_similarity(rights_embs, token_right_embs))
    value_score = tf.reduce_max(euclid_similarity(vals_embs, token_emb))
    left_right_weight = weights[0]
    word_neighbor_weight = weights[1]
    neighbor_score = left_right_weight * left_score + (1. - left_right_weight) * right_score
    token_score = word_neighbor_weight * value_score + (1. - word_neighbor_weight) * neighbor_score
    if is_entity:
        token_score *= 1. + weights[2]
    return token_score
