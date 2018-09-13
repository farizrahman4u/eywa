from ..lang import Document
from ..math import euclid_similarity, vector_sequence_similarity
from ..math import softmax
import numpy as np


class Pattern(object):

    def __init__(self, pattern):
        self._set_pattern(pattern)
        self._get_var_contexts()
        self.weights = np.array([0.5, .1, .1, 1., .05, 0.2])

    def _set_pattern(self, pattern):
        # Converts 'hey there [name: jack, james, !apple, !building]' to 'hey there _eywa_var_name'
        # saves the examples to a dict.
        # No nested [] allowed.

        self._pattern = pattern  # for serialization
        var_to_examples = {}
        y = ''
        buff = ''
        flag = False
        for c in pattern:
            if flag:
                if c == '[':
                    raise Exception('Invalid token \'[\'. Nested [] are not allowed.')
                if c == ']':
                    if ':' in buff:
                        varname, examples = buff.split(':')
                        varname = varname.strip()
                        examples = examples.replace(' ',  '').split(',')
                        if varname in var_to_examples:
                            raise Exception('Multpile definitions for variable {}. '.format(varname) +
                            'Examples should be provided for the first occurence.')
                        positives = []
                        negatives = []
                        p_app = positives.append
                        n_app = negatives.append
                        for e in examples:
                            if e[0] == '!':
                                n_app(Document(e[1:]))
                            else:
                                p_app(Document(e))
                        examples = [positives, negatives]
                        var_to_examples[varname] = examples
                    else:
                        varname = buff
                        if varname not in var_to_examples:
                            var_to_examples[varname] = None
                    y += '_eywa_var_' + varname
                    buff = ''
                    flag = False
                else:
                    buff += c
            else:
                if c == '[':
                    flag = True
                elif c == ']':
                    raise Exception('Unbalanced ]')
                else:
                    y += c
        if flag:
            raise Exception('Unbalanced [')
        self.examples = var_to_examples
        self.vars = list(var_to_examples.keys())
        y = Document(y)
        self.pattern = y
        self.pattern_contexts = self._get_all_contexts(self.pattern)
        self.var_ids = [i for i in range(len(y)) if str(y[i]).startswith('_eywa_var_')]


    def _get_var_contexts(self):
        contexts = {}
        pattern = self.pattern
        for i, t in enumerate(pattern):
            t = str(t)
            if t.startswith('_eywa_var_'):
                var = t[10:]
                left = pattern[:i]
                right = pattern[i + 1:]
                if var in contexts:
                    contexts[var].append([left, right])
                else:
                    contexts[var] = [[left, right]]
        self.contexts = contexts

    def _get_all_contexts(self, x):
        contexts = []
        for i in range(len(x)):
            left = x[: i]
            right = x[i + 1:]
            contexts.append([left, right])
        return contexts

    def _context_similarity(self, c1, c2):
        f = self._similarity
        l1, r1 = c1
        l2, r2 = c2
        lscore = 0.
        nl1 = len(l1)
        nl2 = len(l2)
        if nl1 and nl2:
            lscore = f(l1, l2)
        rscore = 0.
        nr1 = len(r1)
        nr2 = len(r2)
        if nr1 and nr2:
            rscore = f(r1, r2)
        return 0.5 * (lscore + rscore)

    def _similarity(self, x1, x2):
        #if x1 == x2:
        #    return 1
        if len(x1) == 0 or len(x2) == 0:
            return 0.
        weights = self.weights
        w0 = weights[0]
        score1 = lambda : euclid_similarity(x1.embedding, x2.embedding)
        score2 = lambda : np.dot(x1.embedding, x2.embedding)
        score3 = lambda : vector_sequence_similarity(x1.embeddings, x2.embeddings, w0, 'dot')
        score4 = lambda : vector_sequence_similarity(x1.embeddings, x2.embeddings, w0, 'euclid')
        scores = [score1, score2, score3, score4]
        score_weights = weights[1:5]
        score = 0.
        for s, w in zip(scores, score_weights):
            if w > 0.05:
                score += s()
        return score * 0.25

    def __call__(self, input):
        input = Document(input)
        vars = self.vars
        m = len(vars)
        n = len(input)
        f1 = self._context_similarity
        f2 = self._similarity
        examples = self.examples
        input_contexts = self._get_all_contexts(input)
        matrix = np.zeros((m, n))
        contexts = self.contexts
        w = self.weights[5] * 10
        for i in range(m):
            for j in range(n):
                var = vars[i]
                inp_j = input[j : j + 1]
                var_contexts = contexts[var]
                token_context = input_contexts[j]
                scores = [f1(vc, token_context) for vc in var_contexts]
                score = max(scores)
                var_examples = examples[var]
                pos_examples, neg_examples = var_examples
                if var_examples:
                    pos_score = max([f2(ve, inp_j) for ve in pos_examples])
                    neg_score = sum([f2(ve, inp_j) for ve in neg_examples])
                    score += w * pos_score - neg_score
                matrix[i, j] = score
        matrix *= softmax(matrix, 0)
        val_ids = np.argmax(matrix, 1)
        return {vars[i] : str(input[int(val_ids[i])]) for i in range(m)}

    def serialize(self):
        config = {'pattern' : self._pattern}
        config['weights'] = [float(w) for w in self.weights]
        return config

    @classmethod
    def deserialize(cls, config):
        p = cls(config['pattern'])
        p.weights = np.array(config['weights'])
        return p
