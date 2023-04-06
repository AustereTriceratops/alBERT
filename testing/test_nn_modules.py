import unittest

import torch

from bert.attention import dot_product_attention, Attention, MultiheadAttention


class TestNnModules(unittest.TestCase):
    def setUp(self) -> None:
        self.attention = Attention(model_dims=12, key_dims=16, value_dims=8)
        self.multiheadAttention = MultiheadAttention(model_dims=10, key_dims=12, value_dims=8, heads=3)

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

        self.assertEqual(result.shape, input.shape)

if __name__ == "__main__":
    unittest.main()
