import time
import torch
import torch.cuda
import unittest
import logging
import sys
import os

# Add parent directory to sys.path for importing transformer_arch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformer_arch import Config, Transformer

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('llm_debug.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Set seed for reproducibility
torch.manual_seed(42)

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
        self.transformer.eval()  # Ensure inference mode
        torch.manual_seed(42)

    def _measure_time(self, func, *args, num_runs=10, warmup_runs=3):
        """Measure average execution time with GPU synchronization."""
        if device.type == "cuda":
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            # Warmup
            for _ in range(warmup_runs):
                func(*args)
            torch.cuda.synchronize()
            
            # Timed runs
            total_time = 0.0
            for _ in range(num_runs):
                start_event.record()
                func(*args)
                end_event.record()
                torch.cuda.synchronize()
                total_time += start_event.elapsed_time(end_event) / 1000.0 # Convert to seconds
            return total_time / num_runs
        else:
            # CPU fallback using time.perf_counter
            for _ in range(warmup_runs):
                func(*args)
            total_time = 0.0
            for _ in range(num_runs):
                start = time.perf_counter()
                func(*args)
                end = time.perf_counter()
                total_time += (end - start)
            return total_time / num_runs

    def _get_memory_usage(self, func, *args):
        """Measure peak GPU memory usage."""
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            func(*args)
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated() / 1024**2 # MB
        return 0.0 # CPU memory tracking is complex; return 0

    def test_forward_latency_small(self):
        """Test forward pass latency for small inputs."""
        B, T = 2, 10
        input_ids = torch.randint(0, self.config.vocab_size, (B, T)).to(device)
        
        def forward_func():
            with torch.no_grad():
                self.transformer(input_ids)
        
        avg_time = self._measure_time(forward_func)
        logger.info(f"Forward latency (B={B}, T={T}): {avg_time:.4f} seconds")
        self.assertLess(avg_time, 0.1, "Forward pass too slow for small inputs")

    def test_forward_latency_large(self):
        """Test forward pass latency for large inputs."""
        B, T = 8, 256
        input_ids = torch.randint(0, self.config.vocab_size, (B, T)).to(device)
        
        def forward_func():
            with torch.no_grad():
                self.transformer(input_ids)
        
        avg_time = self._measure_time(forward_func)
        logger.info(f"Forward latency (B={B}, T={T}): {avg_time:.4f} seconds")
        self.assertLess(avg_time, 1.0, "Forward pass too slow for large inputs")

    def test_generation_throughput(self):
        """Test generation throughput (tokens per second)."""
        B, T = 4, 64
        input_ids = torch.randint(0, self.config.vocab_size, (B, T)).to(device)
        max_length = T + 10
        
        def generate_func():
            with torch.no_grad():
                self.transformer.generate(input_ids, max_length=max_length, temperature=1.0)
        
        avg_time = self._measure_time(generate_func)
        tokens_generated = B * (max_length - T)
        throughput = tokens_generated / avg_time
        logger.info(f"Generation throughput (B={B}, T={T}, max_length={max_length}): {throughput:.2f} tokens/sec")
        self.assertGreater(throughput, 100, "Generation throughput too low")

    def test_memory_usage(self):
        """Test peak GPU memory usage for large inputs."""
        B, T = 8, 256
        input_ids = torch.randint(0, self.config.vocab_size, (B, T)).to(device)
        
        def forward_func():
            with torch.no_grad():
                self.transformer(input_ids)
        
        memory_mb = self._get_memory_usage(forward_func)
        logger.info(f"Peak memory usage (B={B}, T={T}): {memory_mb:.2f} MB")
        if device.type == "cuda":
            self.assertLess(memory_mb, 5000, "Memory usage too high")

    def test_amp_performance(self):
        """Compare latency with and without AMP."""
        B, T = 4, 64
        input_ids = torch.randint(0, self.config.vocab_size, (B, T)).to(device)
        
        def forward_with_amp():
            with torch.no_grad():
                with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                    self.transformer(input_ids)
        
        def forward_without_amp():
            with torch.no_grad():
                self.transformer(input_ids)
        
        if device.type == "cuda":
            time_with_amp = self._measure_time(forward_with_amp)
            time_without_amp = self._measure_time(forward_without_amp)
            logger.info(f"AMP latency: {time_with_amp:.4f} sec, No AMP: {time_without_amp:.4f} sec")
            self.assertLess(time_with_amp, time_without_amp * 1.1, "AMP not faster than FP32")
        else:
            logger.info("AMP test skipped on CPU")
            self.skipTest("AMP testing requires CUDA")

    def test_kv_cache_efficiency(self):
        """Test decoding latency vs. prefill for KV caching."""
        B, T = 2, 64
        input_ids = torch.randint(0, self.config.vocab_size, (B, T)).to(device)
        single_token = torch.randint(0, self.config.vocab_size, (B, 1)).to(device)
        
        def prefill_func():
            with torch.no_grad():
                self.transformer(input_ids)
        
        def decode_func():
            with torch.no_grad():
                _, past_kv = self.transformer(input_ids)
                self.transformer(single_token, past_key_values=past_kv, kv_cache_offset=T)
        
        prefill_time = self._measure_time(prefill_func)
        decode_time = self._measure_time(decode_func)
        logger.info(f"Prefill time (T={T}): {prefill_time:.4f} sec, Decode time (1 token): {decode_time:.4f} sec")
        self.assertLess(decode_time, prefill_time / 10, "Decoding not faster than prefill with KV cache")

if __name__ == '__main__':
    unittest.main()
