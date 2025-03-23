"""
Base attention implementation for transformer models.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Attention implementation.
    
    This serves as the baseline attention mechanism that will be 
    compared against Radix Attention and MLA.
    """
    
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        use_rotary_emb=False,
    ):
        """
        Initialize Multi-Head Attention.
        
        Args:
            dim (int): Input dimension
            heads (int): Number of attention heads
            dim_head (int): Dimension of each attention head
            dropout (float): Dropout probability
            use_rotary_emb (bool): Whether to use rotary positional embeddings
        """
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.use_rotary_emb = use_rotary_emb
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
        # Initialize Key-Value cache for inference
        self.k_cache = None
        self.v_cache = None
    
    def _apply_rotary_emb(self, x, pos):
        """
        Apply rotary embeddings to input tensors.
        
        Args:
            x (torch.Tensor): Input tensor [batch, seq_len, heads, dim_head]
            pos (torch.Tensor): Position tensor [batch, seq_len]
            
        Returns:
            torch.Tensor: Tensor with rotary embeddings applied
        """
        if not self.use_rotary_emb:
            return x
        
        # This is a placeholder implementation of rotary embeddings
        # A real implementation would include the full rotary calculation
        device = x.device
        seq_len = x.shape[1]
        dim = self.dim_head // 2
        
        # Generate position indices
        if pos is None:
            pos = torch.arange(seq_len, device=device)
            pos = pos.unsqueeze(0).expand(x.shape[0], -1)  # [batch, seq_len]
        
        # Generate frequency bands
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, device=device).float() / dim))
        sinusoid = torch.einsum('b s, d -> b s d', pos.float(), inv_freq)  # [batch, seq_len, dim//2]
        
        # Generate rotations
        sin, cos = torch.sin(sinusoid), torch.cos(sinusoid)
        sin, cos = map(lambda t: torch.repeat_interleave(t, 2, dim=-1), (sin, cos))  # [batch, seq_len, dim]
        
        # Reshape x for rotation
        x1, x2 = x[..., :dim], x[..., dim:]
        
        # Apply rotation
        sin = sin.unsqueeze(2)  # [batch, seq_len, 1, dim]
        cos = cos.unsqueeze(2)  # [batch, seq_len, 1, dim]
        
        rotated = torch.cat([
            x1 * cos - x2 * sin,
            x2 * cos + x1 * sin
        ], dim=-1)
        
        return rotated
    
    def reset_kv_cache(self):
        """Reset the key-value cache."""
        self.k_cache = None
        self.v_cache = None
    
    def forward(
        self, 
        x, 
        mask=None, 
        pos=None, 
        use_cache=False, 
        is_causal=False,
        past_key_value=None,
    ):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor [batch, seq_len, dim]
            mask (torch.Tensor, optional): Attention mask [batch, seq_len, seq_len]
            pos (torch.Tensor, optional): Position tensor [batch, seq_len]
            use_cache (bool): Whether to use KV caching for inference
            is_causal (bool): Whether to use causal masking
            past_key_value (tuple, optional): Cached key and value tensors
            
        Returns:
            torch.Tensor: Output tensor [batch, seq_len, dim]
            tuple: Updated key-value cache if use_cache=True
        """
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h=self.heads), qkv)
        
        # Apply rotary embeddings if specified
        if self.use_rotary_emb:
            q = self._apply_rotary_emb(q, pos)
            k = self._apply_rotary_emb(k, pos)
        
        # Handle KV caching
        if use_cache:
            if past_key_value is not None:
                # Retrieve past keys and values
                past_k, past_v = past_key_value
                k = torch.cat([past_k, k], dim=1)
                v = torch.cat([past_v, v], dim=1)
            
            # Cache current keys and values
            current_key_value = (k, v)
        
        # Calculate attention scores
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        # Apply masking
        if mask is not None:
            mask_value = -torch.finfo(dots.dtype).max
            dots = dots.masked_fill(mask == 0, mask_value)
        
        if is_causal:
            i, j = dots.shape[-2:]
            causal_mask = torch.ones((i, j), device=x.device, dtype=torch.bool).triu(j - i + 1)
            dots = dots.masked_fill(causal_mask, -torch.finfo(dots.dtype).max)
        
        # Attention weights
        attn = F.softmax(dots, dim=-1)
        
        # Compute output
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b n h d -> b n (h d)')
        out = self.to_out(out)
        
        if use_cache:
            return out, current_key_value
        
        return out
