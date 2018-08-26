from ..utils import config_file
import json
import os
import importlib
import sys


_BACKEND = None

_available_backends = ['numpy', 'numba']
_default_backend = 'numba'


def backend():
    return _BACKEND


def set_backend(backend, save_to_config=True):
    global _BACKEND
    assert backend in _available_backends, "Unknown backend : {}".format(backend)
    if not os.path.isfile(config_file):
        config = {}
    else:
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
        except Exception:
                config = {}
                save_to_config = False
    config['math_backend'] = backend
    if save_to_config:
        try:
            with open(config_file, 'w') as f:
                json.dump(config, f)
        except Exception:
            pass
    if _BACKEND is not None and _BACKEND != backend:
        sys.stderr.write("Setting backend to {}.\n".format(backend))
    _BACKEND = backend
    module = 'eywa.math.{}_backend'.format(backend)
    entries = importlib.import_module(module).__dict__
    globals().update(entries)


def _find_backend():
    # 1 - check config file
    # 2 - check env variable
    # 3 - default backend
    backend = None
    save_to_config = True
    if os.path.isfile(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                if config:
                    backend = config.get('math_backend')
                    save_to_config = False
        except Exception:
            save_to_config = False
            pass
    if backend is None:
        backend = os.environ.get('EYWA_MATH_BACKEND')
    if backend is None:
        backend = _default_backend
    set_backend(backend, save_to_config)



_find_backend()

sys.stderr.write('Using {} backend.\n'.format(_BACKEND))
