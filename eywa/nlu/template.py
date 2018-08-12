from ..lang import Document, Token


class Template(object):

    def __init__(self, templates):
        self._templates = templates
        self.templates = []
        if type(templates) in (list, tuple):
            self._list_out = True
            for t in templates:
                self._add_template(t)
        else:
            self._add_template(templates)
            self._list_out = False
        self._create_template_to_index_key_map()

    def _add_template(self, template):
        y = ''
        buff = ''
        flag = False
        for c in template:
            if flag:
                if c == ']':
                    y += '_eywa_var_' + buff
                    buff = ''
                    flag = False
                elif c == '[':
                    raise Exception('Invalid token \'[\'. Nested [] are not allowed.')
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
        y = Document(y)
        self.templates.append(y)

    def _create_template_to_index_key_map(self):
        _template_to_index_key_map = []
        templates = self.templates
        for t in templates:
            ikm = []
            _template_to_index_key_map.append(ikm)
            for j, w in enumerate(t):
                w = str(w)
                if w.startswith('_eywa_var_'):
                    varname = w[10:]
                    ikm.append((j, varname))
        self._template_to_index_key_map = _template_to_index_key_map


    def __call__(self, x):
        t2ikm = self._template_to_index_key_map
        if self._list_out:
            output = []
            for i, t in enumerate(self.templates):
                tokens = t.tokens[:]
                ikm = t2ikm[i]
                for i, k in ikm:
                    xk = x.get(k)
                    if xk is None:
                        raise Exception('Value not provided for variable ' + k)
                    tokens[i] = Token(xk)
                output.append(Document(tokens))
            return output
        else:
            ikm = t2ikm[0]
            tokens = self.templates[0].tokens[:]
            for i, k in ikm:
                xk = x.get(k)
                if xk is None:
                    raise Exception('Value not provided for variable ' + k)   
                tokens[i] = Token(xk)
            return Document(tokens)

    def serialize(self):
        return {'templates': self._templates}

    @classmethod
    def deserialize(cls, config):
        return cls(**config)
