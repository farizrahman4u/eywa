'''
A light weight dict-like wrapper on top of sqlite3 database.
Yes, I am aware of sqlitedict (https://github.com/RaRe-Technologies/sqlitedict).
But we only require a few key type and value type combinations, so no need of all
the pickling.
We have multiple classes for various edge cases - this is to avoid checks in get/set.
'''

import sqlite3
from ast import literal_eval as literal_eval
import sys


py3 = sys.version_info[0] == 3


class _Database(object):

    def __init__(self, file, key_type=str, value_type=str, new=False):
        self.file = file
        self.connection = sqlite3.connect(file)
        self.cursor = self.connection.cursor()
        type_map = {
            str: 'TEXT',
            int: 'INTEGER',
            float: 'REAL'
        }

        allowed_key_types = [str, int]
        if not py3:
            allowed_key_types.append(unicode)  # noqa
            type_map[unicode] = 'TEXT'  # noqa
        if key_type not in allowed_key_types:
            raise Exception(
                'Unsopported key type {}. Supported key types are {}.'.format(
                    str(key_type), str(allowed_key_types)))
        sql_key_type = type_map[key_type]
        sql_value_type = type_map.get(value_type)
        if sql_value_type is None:
            raise Exception(
                'Unsopported value type {}. Supported value types are {}.'.format(
                    str(value_type), str(
                        type_map.keys())))
        if new:
            if self.cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' and name='dict'").fetchone():
                self.cursor.execute("DROP TABLE dict")
            self.cursor.execute(
                "CREATE TABLE dict (k {} PRIMARY KEY, v {})".format(
                    sql_key_type, sql_value_type))
        self.execute = self.cursor.execute
        self.executemany = self.cursor.executemany

    def __contains__(self, key):
        value = self.execute("SELECT v FROM dict WHERE k = ?", (key,)).fetchone()
        return value is not None

    def get(self, key, default=None):
        value = self.execute("SELECT v FROM dict WHERE k = ?", (key,)).fetchone()
        if value is None:
            return default
        return value[0]

    def __getitem__(self, key):
        value = self.execute("SELECT v FROM dict WHERE k = ?", (key,)).fetchone()
        if value is None:
            raise KeyError(key)
        return value[0]

    def __setitem__(self, key, value):
        self.execute("REPLACE INTO dict (k, v) VALUES (?, ?)", (key, value))

    def update(self, values):
        if isinstance(values, dict):
            values = values.items()
        self.executemany("INSERT INTO dict VALUES (?, ?)", values)

    def close(self):
        self.connection.commit()
        self.connection.close()


if not py3:

    class _Py2StrKeyDatabase(_Database):

        def __setitem__(self, key, value):
            self.execute("REPLACE INTO dict (k, v) VALUES (?, ?)",
                         (key.decode('utf-8'), value))

        def __contains__(self, key):
            value = self.execute("SELECT v FROM dict WHERE k = ?",
                                 (key.decode('utf-8'),)).fetchone()
            return value is not None

        def get(self, key, default=None):
            value = self.execute("SELECT v FROM dict WHERE k = ?",
                                 (key.decode('utf-8'),)).fetchone()
            if value is None:
                return default
            return value[0]

        def __getitem__(self, key):
            value = self.execute("SELECT v FROM dict WHERE k = ?",
                                 (key.decode('utf-8'),)).fetchone()
            if value is None:
                raise KeyError(key)
            return value[0]

        def update(self, values):
            if isinstance(values, dict):
                values = values.items()
            values = map(lambda x: (x[0].decode('utf-8'), x[1]), values)
            self.executemany("INSERT INTO dict VALUES (?, ?)", values)

    class _Py2StrValDatabase(_Database):

        def __setitem__(self, key, value):
            self.execute("REPLACE INTO dict (k, v) VALUES (?, ?)",
                         (key, value.decode('utf-8')))

        def get(self, key, default=None):
            value = self.execute("SELECT v FROM dict WHERE k = ?", (key,)).fetchone()
            if value is None:
                return default
            return value[0].encode('utf-8')

        def __getitem__(self, key):
            value = self.execute("SELECT v FROM dict WHERE k = ?", (key,)).fetchone()
            if value is None:
                raise KeyError(key)
            return value[0].encode('utf-8')

        def update(self, values):
            if isinstance(values, dict):
                values = values.items()
            values = map(lambda x: (x[0], x[1].decode('utf-8')), values)
            self.executemany("INSERT INTO dict VALUES (?, ?)", values)

    class _Py2StrKeyStrValDatabase(_Database):

        def __setitem__(self, key, value):
            self.execute("REPLACE INTO dict (k, v) VALUES (?, ?)",
                         (key.decode('utf-8'), value.decode('utf-8')))

        def __contains__(self, key):
            value = self.execute("SELECT v FROM dict WHERE k = ?",
                                 (key.decode('utf-8'),)).fetchone()
            return value is not None

        def get(self, key, default=None):
            value = self.execute("SELECT v FROM dict WHERE k = ?",
                                 (key.decode('utf-8'),)).fetchone()
            if value is None:
                return default
            return value[0].encode('utf-8')

        def __getitem__(self, key):
            value = self.execute("SELECT v FROM dict WHERE k = ?",
                                 (key.decode('utf-8'),)).fetchone()
            if value is None:
                raise KeyError(key)
            return value[0].encode('utf-8')

        def update(self, values):
            if isinstance(values, dict):
                values = values.items()
            values = map(
                lambda x: (
                    x[0].decode('utf-8'),
                    x[1].decode('utf-8')),
                values)
            self.executemany("INSERT INTO dict VALUES (?, ?)", values)

    class _Py2ListValDatabase(_Database):

        def __setitem__(self, key, value):
            value = str(value)
            self.execute("REPLACE INTO dict (k, v) VALUES (?, ?)",
                         (key, value.decode('utf-8')))

        def get(self, key, default=None):
            value = self.execute("SELECT v FROM dict WHERE k = ?", (key,)).fetchone()
            if value is None:
                return default
            return literal_eval(value[0])

        def __getitem__(self, key):
            value = self.execute("SELECT v FROM dict WHERE k = ?", (key,)).fetchone()
            if value is None:
                raise KeyError(key)
            return literal_eval(value[0])

        def update(self, values):
            if isinstance(values, dict):
                values = values.items()
            values = map(lambda x: (x[0], str(x[1]).decode('utf-8')), values)
            self.executemany("INSERT INTO dict VALUES (?, ?)", values)

    class _Py2StrKeyListValDatabase(_Database):

        def __setitem__(self, key, value):
            value = str(value)
            self.execute("REPLACE INTO dict (k, v) VALUES (?, ?)",
                         (key.decode('utf-8'), value.decode('utf-8')))

        def __contains__(self, key):
            value = self.execute("SELECT v FROM dict WHERE k = ?",
                                 (key.decode('utf-8'),)).fetchone()
            return value is not None

        def get(self, key, default=None):
            value = self.execute("SELECT v FROM dict WHERE k = ?",
                                 (key.decode('utf-8'),)).fetchone()
            if value is None:
                return default
            return literal_eval(value[0])

        def __getitem__(self, key):
            value = self.execute("SELECT v FROM dict WHERE k = ?",
                                 (key.decode('utf-8'),)).fetchone()
            if value is None:
                raise KeyError(key)
            return literal_eval(value[0])

        def update(self, values):
            if isinstance(values, dict):
                values = values.items()
            values = map(lambda x: (x[0].decode('utf-8'),
                                    str(x[1]).decode('utf-8')), values)
            self.executemany("INSERT INTO dict VALUES (?, ?)", values)


class _ListDatabase(_Database):

    def __setitem__(self, key, value):
        value = str(value)
        self.execute("REPLACE INTO dict (k, v) VALUES (?, ?)", (key, value))

    def get(self, key, default=None):
        value = self.execute("SELECT v FROM dict WHERE k = ?", (key,)).fetchone()
        if value is None:
            return default
        return literal_eval(value[0])

    def __getitem__(self, key):
        value = self.execute("SELECT v FROM dict WHERE k = ?", (key,)).fetchone()
        if value is None:
            raise KeyError(key)
        return literal_eval(value[0])

    def update(self, values):
        if isinstance(values, dict):
            values = values.items()
        values = map(lambda x: (x[0], str(x[1])), values)
        self.executemany("INSERT INTO dict VALUES (?, ?)", values)


def _cached(_db_class):
    class CachedDatabase(_db_class):
        def __init__(self, *args, **kwargs):
            self.cache = {}
            self.sup = super(self.__class__, self)
            self.get = self.cache.get
            self.sup.__init__(*args, **kwargs)

        def __setitem__(self, key, value):
            # self.sup.__setitem__(key, value)
            self.cache[key] = value

        def __getitem__(self, key):
            return self.cache[key]

        def __contains__(self, key):
            return key in self.cache

        def update(self, values):
            # self.sup.update(values)
            self.cache.update(values)

        def close(self):
            self.sup.update(self.cache)
            self.cache.clear()
            self.sup.close()

    return CachedDatabase


def Database(file, key_type=str, value_type=str, new=False, cached=False):
    if py3:
        if value_type is list:
            value_type = str
            db_class = _ListDatabase
        else:
            db_class = _Database
    else:
        if key_type is str:
            if value_type is str:
                db_class = _Py2StrKeyStrValDatabase
            elif value_type is list:
                value_type = str
                db_class = _Py2StrKeyListValDatabase
            else:
                db_class = _Py2StrKeyDatabase
        else:
            if value_type is str:
                db_class = _Py2StrValDatabase
            elif value_type is list:
                value_type = str
                db_class = _Py2ListValDatabase
            else:
                db_class = _Database
    if cached:
        db_class = _cached(db_class)
    return db_class(file, key_type, value_type, new)
