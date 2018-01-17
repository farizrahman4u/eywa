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

def _str(x):
    try:
        return str(x)
    except:
        return x.encode('utf-8')

class Database(object):

    def __init__(self, file):
        self.file = file
        self.db = dbm.open(file, 'c')

    def __setitem__(self, key, value):
        key = _str(key)
        value = _str(value)
        self.db[key] = value

    def __getitem__(self, key):
        key = _str(key)
        val = self.db[key]
        try:
            val = literal_eval(val)
        except:
            pass
        return val

    def __getattr__(self, attr):
        return getattr(self.db, attr)

    def __contains__(self, key):
        return _str(key) in self.db

    def __len__(self):
        return len(self.db)
