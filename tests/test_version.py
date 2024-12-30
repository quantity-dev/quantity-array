import importlib.metadata

import pint_array


def test_version():
    assert importlib.metadata.version("pint_array") == pint_array.__version__
