import unittest

import torch

from bert.attention import Attention


class TestNnModules(unittest.TestCase):
    def setUp(self) -> None:
        self.attention = Attention(input_channels=12, output_channels=12, hidden_channels=8)

    def test_dot_product_attention_dimensions(self) -> None:
        q = torch.rand((10, 30, 8))
        k = torch.rand((10, 30, 8))
        v = torch.rand((10, 30, 12))
        x = torch.rand((2, 60, 12))

        a = self.attention.dot_product_attention(q, k, v)
        b = self.attention(x)

        self.assertEqual(a.shape, v.shape)
        self.assertEqual(b.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
