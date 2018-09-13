class Signal(object):
    def __init__(self, data, previous=None, meta_data={}, user_id=None, node=None):
        self.data = data
        if user_id is not None:
            if 'user_id' in meta_data and meta_data['user_id'] != user_id:
                raise Exception('You tried to pass a user_id to the'
                                ' Signal constructor but meta_data'
                                ' contains a different user_id.')
            meta_data['user_id'] = user_id
        if type(previous) in (list, tuple):
            for sig in previous:
                for k in sig.meta_data:
                    if k in meta_data:
                        if meta_data[k] != sig.meta_data[k]:
                            raise Exception(
                                'Meta data merge conflict. Multiple values'
                                ' for key ' + k + '.')
                        meta_data[k] = sig.meta_data[k]
            if len(previous) == 1:
                previous = previous[0]
        elif previous:
            for k in previous.meta_data:
                if k in meta_data:
                    if meta_data[k] != sig.meta_data[k]:
                        raise Exception('Meta data merge conflict. Multiple values'
                                        ' for key ' + k + '.')
                meta_data[k] = sig.meta_data[k]
        self.previous = previous
        self.meta_data = meta_data
        self.node = node

    def copy(self, node=None):
        return self.__class__(self.data, self, node=node)
