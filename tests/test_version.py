import importlib.metadata

import quantity_array


def test_version():
    assert importlib.metadata.version("quantity_array") == quantity_array.__version__
