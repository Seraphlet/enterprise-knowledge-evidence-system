"""Smoke tests for the initial package and CLI contract."""

import unittest

import knowledge_system
from knowledge_system.cli import main


class SkeletonSmokeTest(unittest.TestCase):
    def test_package_import_and_cli_start(self) -> None:
        self.assertEqual(knowledge_system.__version__, "0.0.0")
        self.assertEqual(main([]), 0)


if __name__ == "__main__":
    unittest.main()

