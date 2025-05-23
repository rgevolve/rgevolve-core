import unittest

class TestPackage(unittest.TestCase):

    def test_import(self):
        try:
            import rgevolve.tools as tools
        except ImportError:
            self.fail("Importing rgevolve.tools failed.")
        else:
            self.assertIsNotNone(tools)

