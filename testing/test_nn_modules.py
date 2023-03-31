import unittest

import torch

from bert.attention import dot_product_attention, Attention, MultiheadAttention


class TestNnModules(unittest.TestCase):
    def setUp(self) -> None:
        self.attention = Attention(in_channels=12, out_channels=12, hidden_channels=8)
        self.multiheadAttention = MultiheadAttention(in_channels=10, out_channels=10, hidden_channels=5, heads=3)

    def test_dot_product_attention_dimensions(self) -> None:
        q = torch.rand((10, 30, 8))
        k = torch.rand((10, 30, 8))
        v = torch.rand((10, 30, 12))

        result = dot_product_attention(q, k, v)

        self.assertEqual(result.shape, v.shape)

    def test_single_head_attention(self) -> None:
        input = torch.rand((2, 60, 12))

        result = self.attention(input)

        self.assertEqual(result.shape, input.shape)

    def test_multi_head_attention(self) -> None:
        input = torch.rand((2, 60, 10))

        result = self.multiheadAttention(input)

        self.assertEqual(result.shape, (2, 60, 30))

if __name__ == "__main__":
    unittest.main()
