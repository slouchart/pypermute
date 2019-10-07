__all__ = ['test_all']

from unittest import TestSuite, defaultTestLoader
from importlib import import_module


def test_all():

    package_name = __name__
    module_name = 'test_permutations'

    suite = TestSuite()
    loader = defaultTestLoader
    import_module('.'.join([package_name, module_name]), package_name)
    module = globals()[module_name]
    suite.addTest(loader.loadTestsFromModule(module))

    return suite



