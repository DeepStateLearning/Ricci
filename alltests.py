import unittest

from Laplacian import LaplaceTests
from Ricci import RicciTests
from data import DataTests
from tools import ToolsTests

tests = [LaplaceTests, RicciTests, DataTests, ToolsTests]

tests = [unittest.TestLoader().loadTestsFromTestCase(t) for t in tests]
suite = unittest.TestSuite(tests)
unittest.TextTestRunner(verbosity=2).run(suite)
