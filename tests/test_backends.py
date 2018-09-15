from eywa.math.numba_backend import vector_sequence_similarity as nb_sim
from eywa.math.numpy_backend import vector_sequence_similarity as np_sim
import numpy as np
from numpy.testing import assert_allclose


class TestBackend:

    def test_dot(self):
        input_shapes = [[(32, 12), (32, 12)], [(17, 5), (36, 5)], [(2, 3), (5, 3)]]
        inputs = [list(map(np.random.random, shape)) for shape in input_shapes]
        np_outs = [np_sim(x[0], x[1], 0.5, 'dot') for x in inputs]
        nb_outs = [nb_sim(x[0], x[1], 0.5, 'dot') for x in inputs]
        assert_allclose(np_outs, nb_outs, atol=1e-5)

    def test_euclid(self):
        input_shapes = [[(32, 12), (32, 12)], [(17, 5), (36, 5)], [(2, 3), (5, 3)]]
        inputs = [list(map(np.random.random, shape)) for shape in input_shapes]
        np_outs = [np_sim(x[0], x[1], 0.5, 'euclid') for x in inputs]
        nb_outs = [nb_sim(x[0], x[1], 0.5, 'euclid') for x in inputs]
        assert_allclose(np_outs, nb_outs, atol=1e-5)
