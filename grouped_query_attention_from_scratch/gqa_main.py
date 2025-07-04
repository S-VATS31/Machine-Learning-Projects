import math
import torch
import torch.nn.functional as F

# MPS setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {device}")

class GroupedQueryAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, query_groups: int, dropout: float = 0.15):
        """
        Initialize GQA layer.

        Args:
            d_model (int): Dimensionality of the model's input/output representations.
            num_heads (int): Number of attention heads for queries.
            query_groups (int): Number of key/value groups (heads).
        
        Raises:
            ValueError: If `d_model` is not divisible by `num_heads`.
            ValueError: If `num_heads` is not divisible by `query_groups`.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.query_groups = query_groups
        self.dropout = torch.nn.Dropout(p=dropout)
        self.head_dim = d_model // num_heads
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divible by num_heads ({num_heads})")
        if num_heads % query_groups != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by query_groups ({query_groups})")

        # Learnable weight matrices
        self.q_proj = torch.nn.Linear(d_model, self.head_dim * self.num_heads)
        self.k_proj = torch.nn.Linear(d_model, self.head_dim * self.query_groups)
        self.v_proj = torch.nn.Linear(d_model, self.head_dim * self.query_groups)
        self.o_proj = torch.nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):
        """
        Perform forward pass of GQA layer.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, d_model].

        Returns:
            x (torch.Tensor): Output tensor transformed with same shape.

        Raises:
            ValueError: If `x` (input tensor) is not 3 dimensional.
            ValueError: If `D` is not equal to `d_model`.
            ValueError: If `q.shape[-1]` is not equal to `k.shape[-1]`.
            ValueError: If `softmax_attn.shape[-1]` is not equal to `v.shape[-2]`.
            ValueError: If `T' (sequence length) is equal to 0.
        """
        if x.dim() != 3:
            raise ValueError(f"Input tensor, x, must have 3 dimensions, got: {x.dim()} dimensions")
        B, T, D = x.shape
        if D != self.d_model:
            raise ValueError(f"D ({D}) must be equal to d_model ({self.d_model}).")
        # Return empty output tensor if sequence length is 0
        if T == 0:
            o_empty = torch.empty(B, 0, D)
            return o_empty

        # Linear projections
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2) # [B, num_heads, T, head_dim]
        k = self.k_proj(x).reshape(B, T, self.query_groups, self.head_dim).transpose(1, 2) # [B, query_groups, T, head_dim]
        v = self.v_proj(x).reshape(B, T, self.query_groups, self.head_dim).transpose(1, 2) # [B, query_groups, T, head_dim]

        # Expand k and v to match the number of query heads
        heads_per_group = self.num_heads // self.query_groups
        k_expanded = k.unsqueeze(2).expand(B, self.query_groups, heads_per_group, T, self.head_dim).reshape(B, self.num_heads, T, self.head_dim) # [B, num_heads, T, head_dim]
        v_expanded = v.unsqueeze(2).expand(B, self.query_groups, heads_per_group, T, self.head_dim).reshape(B, self.num_heads, T, self.head_dim) # [B, num_heads, T, head_dim]

        # Attention calculation
        if q.shape[-1] != k_expanded.shape[-1]:
            raise ValueError(f"q.shape[-1] ({q.shape[-1]}) must be equal to k_expanded.shape[-1] ({k_expanded.shape[-1]}) for matrix multiplication.")
        attn = torch.matmul(q, k_expanded.transpose(-2, -1)) # [B, num_heads, T, T]
        scaled_attn = attn / math.sqrt(self.head_dim)
        softmax_attn = F.softmax(scaled_attn, dim=-1)
        if softmax_attn.shape[-1] != v_expanded.shape[-2]:
            raise ValueError(f"softmax_attn.shape[-1] ({softmax_attn.shape[-1]}) must be equal to v_expanded.shape[-2] ({v_expanded.shape[-2]}) for matrix multiplication")
        attn_out = torch.matmul(softmax_attn, v_expanded)

        # Concatenate heads
        attn_out = attn_out.transpose(1, 2).reshape(B, T, D)
        o = self.o_proj(attn_out)
        o = self.dropout(o) 
        
        return o
