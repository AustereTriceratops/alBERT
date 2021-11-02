import unittest

import torch

from bert.attention import Attention


class TestNnModules(unittest.TestCase):
    def setUp(self) -> None:
        self.attention = Attention()

    def test_output_dimensions(self) -> None:
        q = torch.rand((10, 30, 8))
        k = torch.rand((10, 30, 8))
        v = torch.rand((10, 30, 12))

        a = self.attention(q, k, v)

        self.assertEqual(a.shape, v.shape)


if __name__ == "__main__":
    unittest.main()
