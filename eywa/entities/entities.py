class Entity(object):
    def __init__(self, data, source_string=None):
        self.data = data
        if source_string is None:
            source_string = str(data)
        self.source_string = source_string

    @property
    def type(self):
        return self.__class__.__name__


class DateTime(Entity):
    pass


class Email(Entity):
    pass


class Url(Entity):
    pass


class PhoneNumber(Entity):
    pass


class Number(Entity):
    pass
