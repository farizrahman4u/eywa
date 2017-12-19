import sys
import ast


py_version = sys.version_info[0]


if py_version == 3:
    import dbm
elif py_version == 2:
    import anydbm as dbm


def literal_eval(x):
    try:
        return ast.literal_eval(x)
    except:
        return x


class Database(object):

    def __init__(self, file):
        self.file = file
        self.db = dbm.open(file, 'c')

    def __setitem__(self, key, value):
        self.db[str(key)] = str(value)

    def __getitem__(self, key):
        return literal_eval(self.db[str(key)])

    def __getattr__(self, attr):
        return getattr(self.db, attr)

    def __contains__(self, key):
        return str(key) in self.db

    def __len__(self):
        return len(self.db)
