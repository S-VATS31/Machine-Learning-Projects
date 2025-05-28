import unittest
import logging
import torch
import sys
import os

# Add parent directory to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformer_arch import RoPE, LayerNorm, MultiHeadAttention, FeedForwardMLP, Config, DecoderBlock, Transformer

# Configure logging once
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

# Set random seed for reproducibility
torch.manual_seed(42)

class TestRoPE(unittest.TestCase):
    def setUp(self):
        self.head_dim = 64
        self.rope = RoPE(head_dim=self.head_dim, base=10000.0).to(device)

    def test_initialization(self):
        self.assertEqual(self.rope.head_dim, self.head_dim)
        self.assertEqual(self.rope.inv_freq.shape, (self.head_dim // 2,))
        self.assertTrue(torch.all(self.rope.inv_freq > 0))

    def test_initialization_odd_head_dim(self):
        with self.assertRaises(ValueError):
            RoPE(head_dim=63).to(device)

    def test_compute_sine_cosine(self):
        T = 10
        sin, cos = self.rope.compute_sine_cosine(T=T)
        self.assertEqual(sin.shape, (1, 1, T, self.head_dim // 2))
        self.assertEqual(cos.shape, (1, 1, T, self.head_dim // 2))
        self.assertTrue(torch.all(sin >= -1.0) and torch.all(sin <= 1.0))
        self.assertTrue(torch.all(cos >= -1.0) and torch.all(cos <= 1.0))
        # Verify sin^2 + cos^2 = 1
        self.assertTrue(torch.allclose(sin**2 + cos**2, torch.ones_like(sin), atol=1e-6))

    def test_compute_sine_cosine_with_offset(self):
        T = 5
        offset = 10
        sin, cos = self.rope.compute_sine_cosine(T=T, offset=offset)
        sin_no_offset, cos_no_offset = self.rope.compute_sine_cosine(T=T + offset)
        self.assertTrue(torch.allclose(sin, sin_no_offset[:, :, offset:offset+T, :], atol=1e-6))
        self.assertTrue(torch.allclose(cos, cos_no_offset[:, :, offset:offset+T, :], atol=1e-6))

    def test_create_rotary(self):
        B, T, num_heads = 2, 10, 4
        x = torch.randn(B, T, num_heads, self.head_dim).to(device)
        sin, cos = self.rope.compute_sine_cosine(T=T)
        x_rotated = self.rope.create_rotary(x, sin, cos)
        self.assertEqual(x_rotated.shape, (B, T, num_heads, self.head_dim))
        self.assertFalse(torch.isinf(x_rotated).any())
        self.assertFalse(torch.isnan(x_rotated).any())

    def test_forward(self):
        B, T, d_model = 2, 10, 128
        x = torch.randn(B, T, d_model).to(device)
        rope = RoPE(head_dim=32).to(device)
        x_rotated = rope(x)
        self.assertEqual(x_rotated.shape, (B, T, d_model))
        self.assertFalse(torch.isinf(x_rotated).any())
        self.assertFalse(torch.isnan(x_rotated).any())

    def test_forward_invalid_d_model(self):
        rope = RoPE(head_dim=30).to(device)
        x = torch.randn(2, 10, 128).to(device)
        with self.assertRaises(ValueError):
            rope(x)

class TestLayerNorm(unittest.TestCase):
    def setUp(self):
        self.d_model = 512
        self.layer_norm = LayerNorm(normalized_shape=self.d_model).to(device)

    def test_initialization(self):
        self.assertEqual(self.layer_norm.gamma.shape, (self.d_model,))
        self.assertEqual(self.layer_norm.beta.shape, (self.d_model,))
        self.assertTrue(torch.all(self.layer_norm.gamma == 1.0))
        self.assertTrue(torch.all(self.layer_norm.beta == 0.0))
        self.assertEqual(self.layer_norm.eps, 1e-6)

    def test_forward(self):
        B, T = 2, 10
        x = torch.randn(B, T, self.d_model).to(device)
        output = self.layer_norm(x)
        self.assertEqual(output.shape, (B, T, self.d_model))
        # Verify normalized mean is close to 0
        mean = output.mean(dim=-1, keepdim=True)
        self.assertTrue(torch.allclose(mean, torch.zeros_like(mean), atol=1e-5))
        # Verify normalized variance is close to 1
        var = output.var(dim=-1, unbiased=False, keepdim=True)
        self.assertTrue(torch.allclose(var, torch.ones_like(var), atol=1e-5))

    def test_numerical_stability(self):
        x = torch.full((2, 10, self.d_model), 1e10).to(device)
        output = self.layer_norm(x)
        self.assertFalse(torch.isinf(output).any())
        self.assertFalse(torch.isnan(output).any())

class TestMultiHeadAttention(unittest.TestCase):
    def setUp(self):
        self.d_model = 512
        self.num_heads = 8
        self.mha = MultiHeadAttention(d_model=self.d_model, num_heads=self.num_heads, dropout=0.0).to(device)

    def test_initialization(self):
        self.assertEqual(self.mha.d_k, self.d_model // self.num_heads)
        self.assertEqual(self.mha.W_Q.in_features, self.d_model)
        self.assertEqual(self.mha.W_Q.out_features, self.d_model)
        self.assertIsInstance(self.mha.rope, RoPE)
        self.assertEqual(self.mha.rope.head_dim, self.d_model // self.num_heads)

    def test_initialization_invalid_d_model(self):
        with self.assertRaises(ValueError):
            MultiHeadAttention(d_model=510, num_heads=8).to(device)

    def test_forward(self):
        B, T = 2, 10
        x = torch.randn(B, T, self.d_model).to(device)
        output, weights, present_kv = self.mha(x, causal=True)
        self.assertEqual(output.shape, (B, T, self.d_model))
        self.assertEqual(weights.shape, (B, self.num_heads, T, T))
        self.assertEqual(present_kv[0].shape, (B, self.num_heads, T, self.d_model // self.num_heads))
        self.assertEqual(present_kv[1].shape, (B, self.num_heads, T, self.d_model // self.num_heads))
        self.assertFalse(torch.isinf(output).any())
        self.assertFalse(torch.isnan(output).any())
        # Verify attention weights sum to 1
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(B, self.num_heads, T).to(device), atol=1e-5))

    def test_forward_with_kv_cache(self):
        B, T = 2, 5
        x = torch.randn(B, T, self.d_model).to(device)
        _, _, past_kv = self.mha(x, causal=True)
        x_new = torch.randn(B, 1, self.d_model).to(device)
        output, weights, new_kv = self.mha(x_new, past_kv=past_kv, causal=True, kv_cache_offset=T)
        self.assertEqual(output.shape, (B, 1, self.d_model))
        self.assertEqual(weights.shape, (B, self.num_heads, 1, T + 1))
        self.assertEqual(new_kv[0].shape, (B, self.num_heads, T + 1, self.d_model // self.num_heads))
        self.assertEqual(new_kv[1].shape, (B, self.num_heads, T + 1, self.d_model // self.num_heads))

    def test_forward_with_padding_mask(self):
        B, T = 2, 10
        x = torch.randn(B, T, self.d_model).to(device)
        padding_mask = torch.ones(B, T, dtype=torch.int, device=device)
        padding_mask[:, -2:] = 0  # Mask last two positions
        output, weights, _ = self.mha(x, padding_mask=padding_mask, causal=True)
        self.assertEqual(output.shape, (B, T, self.d_model))
        self.assertEqual(weights.shape, (B, self.num_heads, T, T))
        # Verify masked positions have zero weight
        self.assertTrue(torch.all(weights[:, :, :, -2:] == 0))

    def test_forward_empty_sequence(self):
        B = 2
        x = torch.randn(B, 0, self.d_model).to(device)
        output, weights, present_kv = self.mha(x, causal=True)
        self.assertEqual(output.shape, (B, 0, self.d_model))
        self.assertEqual(weights.shape, (B, self.num_heads, 0, 0))
        self.assertIsNone(present_kv[0])
        self.assertIsNone(present_kv[1])

class TestFeedForwardMLP(unittest.TestCase):
    def setUp(self):
        self.d_model = 512
        self.d_ffn = 2048
        self.mlp = FeedForwardMLP(d_model=self.d_model, d_ffn=self.d_ffn, dropout=0.0).to(device)

    def test_initialization(self):
        self.assertEqual(self.mlp.linear1.in_features, self.d_model)
        self.assertEqual(self.mlp.linear1.out_features, self.d_ffn)
        self.assertEqual(self.mlp.linear2.in_features, self.d_ffn)
        self.assertEqual(self.mlp.linear2.out_features, self.d_model)

    def test_forward(self):
        B, T = 2, 10
        x = torch.randn(B, T, self.d_model).to(device)
        output = self.mlp(x)
        self.assertEqual(output.shape, (B, T, self.d_model))
        self.assertFalse(torch.isinf(output).any())
        self.assertFalse(torch.isnan(output).any())

class TestDecoderBlock(unittest.TestCase):
    def setUp(self):
        self.d_model = 512
        self.d_ffn = 2048
        self.num_heads = 8
        self.decoder = DecoderBlock(d_model=self.d_model, d_ffn=self.d_ffn, num_heads=self.num_heads, dropout=0.0).to(device)

    def test_initialization(self):
        self.assertIsInstance(self.decoder.MHA, MultiHeadAttention)
        self.assertIsInstance(self.decoder.MLP, FeedForwardMLP)
        self.assertIsInstance(self.decoder.layer_norm1, LayerNorm)
        self.assertIsInstance(self.decoder.layer_norm2, LayerNorm)

    def test_forward(self):
        B, T = 2, 10
        x = torch.randn(B, T, self.d_model).to(device)
        output, present_kv = self.decoder(x)
        self.assertEqual(output.shape, (B, T, self.d_model))
        self.assertEqual(present_kv[0].shape, (B, self.num_heads, T, self.d_model // self.num_heads))
        self.assertEqual(present_kv[1].shape, (B, self.num_heads, T, self.d_model // self.num_heads))
        self.assertFalse(torch.isinf(output).any())
        self.assertFalse(torch.isnan(output).any())

    def test_forward_with_kv_cache(self):
        B, T = 2, 5
        x = torch.randn(B, T, self.d_model).to(device)
        _, past_kv = self.decoder(x)
        x_new = torch.randn(B, 1, self.d_model).to(device)
        padding_mask = torch.ones(B, T + 1, dtype=torch.int, device=device)
        output, new_kv = self.decoder(x_new, past_kv=past_kv, padding_mask=padding_mask, kv_cache_offset=T)
        self.assertEqual(output.shape, (B, 1, self.d_model))
        self.assertEqual(new_kv[0].shape, (B, self.num_heads, T + 1, self.d_model // self.num_heads))
        self.assertEqual(new_kv[1].shape, (B, self.num_heads, T + 1, self.d_model // self.num_heads))

class TestTransformer(unittest.TestCase):
    def setUp(self):
        self.config = Config(
            block_size=512,
            vocab_size=4096,
            num_layers=2,
            num_heads=8,
            d_model=512,
            dropout=0.0,
            d_ffn=2048,
            pad_token_id=0
        )
        self.transformer = Transformer(self.config).to(device)

    def test_initialization(self):
        self.assertEqual(len(self.transformer.blocks), self.config.num_layers)
        self.assertEqual(self.transformer.token_embedding.num_embeddings, self.config.vocab_size)
        self.assertEqual(self.transformer.token_embedding.embedding_dim, self.config.d_model)
        self.assertEqual(self.transformer.head.out_features, self.config.vocab_size)

    def test_forward(self):
        B, T = 2, 10
        input_ids = torch.randint(0, self.config.vocab_size, (B, T)).to(device)
        logits, new_kv = self.transformer(input_ids)
        self.assertEqual(logits.shape, (B, T, self.config.vocab_size))
        self.assertEqual(len(new_kv), self.config.num_layers)
        for k, v in new_kv:
            self.assertEqual(k.shape, (B, self.config.num_heads, T, self.config.d_model // self.config.num_heads))
            self.assertEqual(v.shape, (B, self.config.num_heads, T, self.config.d_model // self.config.num_heads))
        self.assertFalse(torch.isinf(logits).any())
        self.assertFalse(torch.isnan(logits).any())

    def test_generate(self):
        B, T = 2, 5
        input_ids = torch.randint(0, self.config.vocab_size, (B, T)).to(device)
        max_length = T + 5
        generated = self.transformer.generate(input_ids, max_length=max_length, temperature=1.0)
        self.assertEqual(generated.shape, (B, max_length))
        self.assertTrue(torch.all(generated[:, :T] == input_ids))
        self.assertTrue(torch.all(generated >= 0) and torch.all(generated < self.config.vocab_size))

    def test_generate_with_padding(self):
        B, T = 2, 5
        input_ids = torch.randint(1, self.config.vocab_size, (B, T)).to(device)
        input_ids[:, -2:] = self.config.pad_token_id
        max_length = T + 5
        generated = self.transformer.generate(input_ids, max_length=max_length, temperature=1.0)
        self.assertEqual(generated.shape, (B, max_length))
        self.assertTrue(torch.all(generated[:, :T] == input_ids))

    def test_generate_invalid_temperature(self):
        B, T = 2, 5
        input_ids = torch.randint(0, self.config.vocab_size, (B, T)).to(device)
        with self.assertRaises(ValueError):
            self.transformer.generate(input_ids, max_length=10, temperature=0.0)

    def test_generate_top_k(self):
        B, T = 2, 5
        input_ids = torch.randint(0, self.config.vocab_size, (B, T)).to(device)
        generated = self.transformer.generate(input_ids, max_length=T + 5, temperature=1.0, top_k=5)
        self.assertEqual(generated.shape, (B, T + 5))

    def test_generate_top_p(self):
        B, T = 2, 5
        input_ids = torch.randint(0, self.config.vocab_size, (B, T)).to(device)
        generated = self.transformer.generate(input_ids, max_length=T + 5, temperature=1.0, top_p=0.9)
        self.assertEqual(generated.shape, (B, T + 5))

if __name__ == '__main__':
    unittest.main()
