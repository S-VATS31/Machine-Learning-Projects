import unittest
import logging
import math
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformer_arch import RoPE

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('llm_debug.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

torch.manual_seed(42)

class TestRoPE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up class-level resources before any tests in this class run."""
        print(f"\n--- Running RoPE tests on device: {device} ---")

    def setUp(self):
        """Set up resources before each test method runs."""
        self.head_dim = 64
        self.base = 10000.0

    def test_01_rope_initialization(self):
        """Test the initialization of the RoPE module."""
        logger.info("Running test_01_rope_initialization")
        rope = RoPE(self.head_dim, base=self.base)
        self.assertEqual(rope.head_dim, self.head_dim, "head_dim mismatch")
        self.assertIsInstance(rope.inv_freq, torch.Tensor, "inv_freq is not a tensor")
        self.assertEqual(rope.inv_freq.shape, (self.head_dim // 2,), "inv_freq shape mismatch")

        expected_inv_freq = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.assertTrue(torch.allclose(rope.inv_freq, expected_inv_freq.to(device)), "inv_freq values mismatch")
        logger.info("test_01_rope_initialization passed.")

    def test_02_rope_initialization_odd_head_dim_raises_error(self):
        """Test that RoPE initialization raises ValueError for odd head_dim."""
        logger.info("Running test_02_rope_initialization_odd_head_dim_raises_error")
        odd_head_dim = 65
        with self.assertRaisesRegex(ValueError, "head_dim must be divisible by 2"):
            RoPE(odd_head_dim)
        logger.info("test_02_rope_initialization_odd_head_dim_raises_error passed.")

    def test_03_compute_sine_cosine_shapes(self):
        """Test the shapes of sine and cosine tensors computed by RoPE."""
        logger.info("Running test_03_compute_sine_cosine_shapes")
        rope = RoPE(self.head_dim)
        T = 10
        sin, cos = rope.compute_sine_cosine(T)
        self.assertEqual(sin.shape, (1, 1, T, self.head_dim // 2), "sin shape mismatch")
        self.assertEqual(cos.shape, (1, 1, T, self.head_dim // 2), "cos shape mismatch")
        self.assertEqual(sin.device, device, "sin device mismatch")
        self.assertEqual(cos.device, device, "cos device mismatch")
        logger.info("test_03_compute_sine_cosine_shapes passed.")

    def test_04_compute_sine_cosine_values(self):
        """Test the numerical values of sine and cosine tensors."""
        logger.info("Running test_04_compute_sine_cosine_values")
        test_head_dim = 16
        test_base = 100.0
        rope = RoPE(test_head_dim, base=test_base)
        T = 5
        sin, cos = rope.compute_sine_cosine(T)

        pos = 2
        dim_idx = 0
        expected_inv_freq_0 = 1.0 / (test_base ** (torch.tensor(0.0) / test_head_dim))
        expected_theta_0 = pos * expected_inv_freq_0.item()
        self.assertAlmostEqual(sin[0, 0, pos, dim_idx].item(), math.sin(expected_theta_0), places=6, msg="sin value mismatch at pos 2, dim 0")
        self.assertAlmostEqual(cos[0, 0, pos, dim_idx].item(), math.cos(expected_theta_0), places=6, msg="cos value mismatch at pos 2, dim 0")

        pos = 3
        dim_idx = 1 
        expected_inv_freq_1 = 1.0 / (test_base ** (torch.tensor(2.0) / test_head_dim))
        expected_theta_1 = pos * expected_inv_freq_1.item()
        self.assertAlmostEqual(sin[0, 0, pos, dim_idx].item(), math.sin(expected_theta_1), places=6, msg="sin value mismatch at pos 3, dim 1")
        self.assertAlmostEqual(cos[0, 0, pos, dim_idx].item(), math.cos(expected_theta_1), places=6, msg="cos value mismatch at pos 3, dim 1")
        logger.info("test_04_compute_sine_cosine_values passed.")

    def test_05_create_rotary_shapes(self):
        """Test the output shape of the create_rotary method."""
        logger.info("Running test_05_create_rotary_shapes")
        test_head_dim = 32
        num_heads = 4
        d_model = num_heads * test_head_dim
        rope = RoPE(test_head_dim)
        B, T = 2, 10
        x = torch.randn(B, T, num_heads, test_head_dim).to(device)
        sin, cos = rope.compute_sine_cosine(T)
        rotated_x = rope.create_rotary(x, sin, cos)
        self.assertEqual(rotated_x.shape, x.shape, "create_rotary output shape mismatch")
        logger.info("test_05_create_rotary_shapes passed.")

    def test_06_create_rotary_values_simple_case(self):
        """Test the numerical correctness of create_rotary with a simple input."""
        logger.info("Running test_06_create_rotary_values_simple_case")
        test_head_dim = 4
        num_heads = 1
        rope = RoPE(test_head_dim, base=1.0)
        B, T = 1, 1 
        x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]], device=device)

        sin, cos = rope.compute_sine_cosine(T)

        rotated_x = rope.create_rotary(x, sin, cos)

        expected_rotated_x = x
        self.assertTrue(torch.allclose(rotated_x, expected_rotated_x, atol=1e-6), "create_rotary values mismatch for simple case (no rotation)")
        logger.info("test_06_create_rotary_values_simple_case passed.")

    def test_07_rope_forward_shapes(self):
        """Test the output shape of the RoPE forward pass."""
        logger.info("Running test_07_rope_forward_shapes")
        test_head_dim = 32
        d_model = 256 
        rope = RoPE(test_head_dim)
        B, T = 4, 10
        x = torch.randn(B, T, d_model).to(device) 
        rotated_x = rope.forward(x)
        self.assertEqual(rotated_x.shape, x.shape, "RoPE forward output shape mismatch")
        logger.info("test_07_rope_forward_shapes passed.")

    def test_08_rope_forward_d_model_not_divisible_by_head_dim_raises_error(self):
        """Test that RoPE forward raises ValueError if d_model is not divisible by head_dim."""
        logger.info("Running test_08_rope_forward_d_model_not_divisible_by_head_dim_raises_error")
        test_head_dim = 32
        d_model = 250 
        rope = RoPE(test_head_dim)
        B, T = 4, 10
        x = torch.randn(B, T, d_model).to(device)
        with self.assertRaisesRegex(ValueError, f"d_model \\({d_model}\\) must be divisible by head_dim \\({test_head_dim}\\)"):
            rope.forward(x)
        logger.info("test_08_rope_forward_d_model_not_divisible_by_head_dim_raises_error passed.")

    def test_09_rope_forward_offset(self):
        """Test that RoPE forward works correctly with an offset."""
        logger.info("Running test_09_rope_forward_offset")
        test_head_dim = 64
        d_model = 256
        rope = RoPE(test_head_dim)
        B, T = 2, 5
        offset = 3
        x = torch.randn(B, T, d_model).to(device)
        rotated_x_offset = rope.forward(x, offset=offset)

        self.assertEqual(rotated_x_offset.shape, x.shape, "RoPE forward with offset output shape mismatch")

        x_full_length = torch.randn(B, T + offset, d_model).to(device) 
        rotated_x_full_length = rope.forward(x_full_length, offset=0)

        x_for_offset_test = torch.randn(B, T, d_model).to(device)
        rotated_with_offset = rope.forward(x_for_offset_test, offset=offset)

        q_first_token_head0 = x_for_offset_test[0, 0, :test_head_dim].view(1, 1, 1, test_head_dim) 
        sin_at_offset, cos_at_offset = rope.compute_sine_cosine(T=1, offset=offset)

        expected_rotated_first_token_head0 = rope.create_rotary(q_first_token_head0, sin_at_offset, cos_at_offset).view(test_head_dim)

        actual_rotated_first_token_head0 = rotated_with_offset[0, 0, :test_head_dim]

        self.assertTrue(torch.allclose(actual_rotated_first_token_head0, expected_rotated_first_token_head0, atol=1e-5),
                        "RoPE forward with offset has incorrect rotation for first token.")
        logger.info("test_09_rope_forward_offset passed.")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
