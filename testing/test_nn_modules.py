import unittest

import torch

from bert.attention import dot_product_attention, Attention


class TestNnModules(unittest.TestCase):
    def setUp(self) -> None:
        self.attention = Attention(in_channels=12, out_channels=12, hidden_channels=8)

    def test_dot_product_attention_dimensions(self) -> None:
        q = torch.rand((10, 30, 8))
        k = torch.rand((10, 30, 8))
        v = torch.rand((10, 30, 12))

        a = dot_product_attention(q, k, v)

        self.assertEqual(a.shape, v.shape)

    def test_single_head_attention(self) -> None:
        x = torch.rand((2, 60, 12))

        b = self.attention(x)

        self.assertEqual(b.shape, x.shape)



if __name__ == "__main__":
    unittest.main()
