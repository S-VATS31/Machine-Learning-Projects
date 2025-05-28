@dataclass
class Config:
    block_size: int = 512
    vocab_size: int = 4096
    num_layers: int = 12
    num_heads: int = 8
    d_model: int = 512
    dropout: float = 0.1
    d_ffn: int = 2048
    pad_token_id: int = 0
