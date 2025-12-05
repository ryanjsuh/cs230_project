"""
Decoder-only Transformer for prediction market forecasting.

Architecture alignment with Google's TimesFM:
Patch-based tokenization with ResidualBlock tokenizer, RevIN (Reversible Instance Normalization) for input normalization, Stacked causal Transformer decoder blocks, Output patch length can exceed input patch length (4:1 ratio in TimesFM), Mask concatenation with inputs in tokenizer

Domain-specific adaptations for prediction markets:
Sigmoid activation to bound outputs to [0, 1] probability range, Hours-to-resolution auxiliary input, Learned category embeddings for market type conditioning
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass

from model.config import ModelConfig


# KV cache for efficient autoregressive decoding
@dataclass
class DecodeCache:
    next_index: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor


# Reversible Instance Normalization
class RevIN(nn.Module):
    # Normalizes inputs using running mean/std
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    
    # Apply or reverse instance normalization
    def forward(
        self, 
        x: torch.Tensor, 
        mu: torch.Tensor, 
        sigma: torch.Tensor, 
        reverse: bool = False
    ) -> torch.Tensor:
        # Apply or reverse instance normalization
        sigma = sigma.clamp(min=self.eps)
        
        if reverse:
            # Denormalize: x * sigma + mu
            return x * sigma + mu
        else:
            # Normalize: (x - mu) / sigma
            return (x - mu) / sigma


# Residual MLP block for patch embedding
class ResidualBlock(nn.Module):
    def __init__(
        self, 
        in_dim: int, 
        hidden_dim: int, 
        out_dim: int, 
        dropout: float = 0.1,
        use_layer_norm: bool = True
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.norm = nn.LayerNorm(out_dim)
        
        # Projection for residual if dimensions don't match
        self.residual_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = x + residual
        if self.use_layer_norm:
            x = self.norm(x)
        return x


# Tokenizer that converts price patches to embeddings
class PatchTokenizer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Input: (batch, num_patches, patch_len * (n_price_features + n_aux_features + 1 for mask))
        input_dim = config.input_patch_len * (config.n_price_features + config.n_aux_features + 1)
        
        # Two-layer residual block
        self.block1 = ResidualBlock(input_dim, config.d_ff, config.d_model, config.dropout)
        self.block2 = ResidualBlock(config.d_model, config.d_ff, config.d_model, config.dropout)
    
    # Tokenize patches
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, num_patches, patch_len, n_features = x.shape
        
        mask_float = mask.float().unsqueeze(-1)  # (batch, num_patches, patch_len, 1)
        x_with_mask = torch.cat([x, mask_float], dim=-1)  # (batch, num_patches, patch_len, n_features + 1)
        
        x_flat = x_with_mask.view(batch, num_patches, -1)  # (batch, num_patches, patch_len * (n_features + 1))
        
        # Apply residual blocks
        x = self.block1(x_flat)
        x = self.block2(x)
        
        return x


# Learnable positional encoding for patch positions
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Learnable positional embeddings
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    
    # Add positional encoding to input
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


# Learnable embeddings for market categories
class CategoryEmbedding(nn.Module):
    def __init__(self, n_categories: int, embed_dim: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(n_categories, embed_dim)
        self.proj = nn.Linear(embed_dim, d_model)
    
    # Get category embeddings projected to model dimension
    def forward(self, category_ids: torch.Tensor) -> torch.Tensor:
        embed = self.embedding(category_ids)
        return self.proj(embed)


# Multi-head causal self-attention with optional KV caching
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    # Causal self-attention with optional KV caching
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        decode_cache: Optional[DecodeCache] = None,
    ) -> Tuple[torch.Tensor, Optional[DecodeCache]]:
        batch, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Handle KV caching for autoregressive decoding
        new_cache = None
        if decode_cache is not None:
            # Append new K, V to cache
            cache_k = decode_cache.key
            cache_v = decode_cache.value
            
            # Update cache (simplified for now)
            k = torch.cat([cache_k, k], dim=2)
            v = torch.cat([cache_v, v], dim=2)
            
            new_cache = DecodeCache(
                next_index=decode_cache.next_index + seq_len,
                key=k,
                value=v,
            )
        
        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Apply causal mask
        kv_len = k.size(2)
        if mask is None:
            # Create causal mask
            causal_mask = torch.triu(
                torch.ones(seq_len, kv_len, device=x.device, dtype=torch.bool), 
                diagonal=kv_len - seq_len + 1
            )
            attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        else:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        
        return self.out_proj(out), new_cache


# Transformer decoder block with pre-norm
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        decode_cache: Optional[DecodeCache] = None,
    ) -> Tuple[torch.Tensor, Optional[DecodeCache]]:
        # Pre-norm self-attention
        attn_out, new_cache = self.attn(self.norm1(x), mask, decode_cache)
        x = x + attn_out
        # Pre-norm FFN
        x = x + self.ffn(self.norm2(x))
        return x, new_cache


# Output projection head for price predictions
class OutputProjection(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # ResidualBlock for output projection (matching TimesFM)
        self.proj = ResidualBlock(
            config.d_model, 
            config.d_ff, 
            config.output_patch_len,
            config.dropout,
            use_layer_norm=False
        )
        
        self.use_sigmoid = config.use_sigmoid
    
    # Project to price predictions
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.proj(x)
        if self.use_sigmoid:
            out = torch.sigmoid(out)
        return out


# TimesFM-inspired decoder-only Transformer for prediction market forecasting
class PredictionMarketTimesFM(nn.Module):
    """    
    Architecture:
    1. PatchTokenizer: ResidualBlock that tokenizes [prices, aux_features, mask]
    2. RevIN: Reversible instance normalization for input standardization
    3. PositionalEncoding: Learnable patch position embeddings
    4. CategoryEmbedding: Domain-specific market category conditioning
    5. TransformerBlocks: Stack of pre-norm causal decoder blocks with KV caching
    6. OutputProjection: ResidualBlock projecting to output patch predictions
    7. Sigmoid activation: Bounds outputs to [0, 1] for probability prediction
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # RevIN for normalization
        self.revin = RevIN()
        
        # Patch tokenizer
        self.tokenizer = PatchTokenizer(config)
        
        # Category embedding
        self.category_embed = CategoryEmbedding(
            config.n_categories, 
            config.category_embed_dim,
            config.d_model
        )
        
        # Positional encoding
        max_patches = config.context_patches + config.horizon_patches + 10  # Buffer
        self.pos_encoding = PositionalEncoding(
            config.d_model, 
            max_len=max_patches, 
            dropout=config.dropout
        )
        
        # Stacked transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(config.d_model)
        
        # Output projection
        self.output_proj = OutputProjection(config)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    # Compute running mean/std per patch for RevIN
    def _compute_patch_stats(
        self, 
        patches: torch.Tensor, 
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prices = patches[..., 0]  # (batch, num_patches, patch_len)
        
        valid_mask = ~mask 
        valid_count = valid_mask.sum(dim=-1, keepdim=True).clamp(min=1)  # (batch, num_patches, 1)
        
        # Masked mean
        masked_prices = prices * valid_mask.float()
        mu = masked_prices.sum(dim=-1, keepdim=True) / valid_count  # (batch, num_patches, 1)
        
        # Masked std
        sq_diff = ((prices - mu) * valid_mask.float()) ** 2
        variance = sq_diff.sum(dim=-1, keepdim=True) / valid_count
        sigma = torch.sqrt(variance + 1e-8).clamp(min=1e-4)  
        
        return mu.unsqueeze(-1), sigma.unsqueeze(-1)  # (batch, num_patches, 1, 1)
    
    # Forward pass
    def forward(
        self,
        price_patches: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        category_ids: Optional[torch.Tensor] = None,
        decode_caches: Optional[list] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[list]]:
        batch_size, num_patches, patch_len, n_features = price_patches.shape
        
        # Create default mask if not provided
        if mask is None:
            mask = torch.zeros(batch_size, num_patches, patch_len, 
                             dtype=torch.bool, device=price_patches.device)
        
        # Compute patch statistics for RevIN
        mu, sigma = self._compute_patch_stats(price_patches, mask)
        
        # Apply RevIN normalization to price feature
        normed_patches = price_patches.clone()
        normed_patches[..., 0:1] = self.revin(
            price_patches[..., 0:1], 
            mu, sigma, 
            reverse=False
        )
    
        # Zero out masked positions
        normed_patches = torch.where(
            mask.unsqueeze(-1).expand_as(normed_patches),
            torch.zeros_like(normed_patches),
            normed_patches
        )
        
        # Tokenize patches
        x = self.tokenizer(normed_patches, mask)  # (batch, num_patches, d_model)
        
        # Add category embedding if provided
        if category_ids is not None:
            cat_embed = self.category_embed(category_ids)  # (batch, d_model)
            x = x + cat_embed.unsqueeze(1)  # Broadcast to all patches
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Initialize decode caches if needed
        if decode_caches is None:
            decode_caches = [None] * len(self.transformer_layers)
        
        # Pass through transformer layers
        new_caches = []
        for i, layer in enumerate(self.transformer_layers):
            x, new_cache = layer(x, None, decode_caches[i])
            new_caches.append(new_cache)
        
        # Final norm
        x = self.final_norm(x)
        
        # Output projection
        predictions = self.output_proj(x)  # (batch, num_patches, output_patch_len)
        
        return predictions, (mu.squeeze(-1), sigma.squeeze(-1)), new_caches
    
    # Autoregressive generation of future patches
    @torch.no_grad()
    def generate(
        self,
        context_patches: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        category_ids: Optional[torch.Tensor] = None,
        num_steps: int = 1,
    ) -> torch.Tensor:
        self.eval()
        device = context_patches.device
        batch_size = context_patches.size(0)
        
        # Prefill with context
        predictions, (mu, sigma), decode_caches = self.forward(
            context_patches, 
            context_mask, 
            category_ids
        )
        
        # Get prediction for last context patch
        all_outputs = [predictions[:, -1:, :]]  # (batch, 1, output_patch_len)
        
        # Autoregressive decoding for additional steps
        if num_steps > 1:
            # Use last prediction as input for next step
            last_pred = predictions[:, -1, :]  # (batch, output_patch_len)
            
            # Reshape prediction to patches
            for step in range(num_steps - 1):
                # output_patch_len -> m input patches where m = output_patch_len // input_patch_len
                m = self.config.output_patch_len // self.config.input_patch_len
                new_patches = last_pred.view(batch_size, m, self.config.input_patch_len)
                
                new_features = torch.zeros(
                    batch_size, m, self.config.input_patch_len, 
                    self.config.n_price_features + self.config.n_aux_features,
                    device=device
                )
                new_features[..., 0] = new_patches 
                
                # No mask for generated patches
                new_mask = torch.zeros(
                    batch_size, m, self.config.input_patch_len,
                    dtype=torch.bool, device=device
                )
                
                # Forward pass with caching
                predictions, _, decode_caches = self.forward(
                    new_features,
                    new_mask,
                    category_ids,
                    decode_caches
                )
                
                all_outputs.append(predictions[:, -1:, :])
                last_pred = predictions[:, -1, :]
        
        return torch.cat(all_outputs, dim=1)  # (batch, num_steps, output_patch_len)
    
    # Count trainable parameters
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Factory function to create model
def create_model(config: ModelConfig) -> PredictionMarketTimesFM:
    model = PredictionMarketTimesFM(config)
    print(f"Created model with {model.count_parameters():,} parameters")
    print(f"  d_model: {config.d_model}")
    print(f"  n_layers: {config.n_layers}")
    print(f"  n_heads: {config.n_heads}")
    print(f"  input_patch_len: {config.input_patch_len}")
    print(f"  output_patch_len: {config.output_patch_len}")
    print(f"  context_length: {config.context_length}")
    print(f"  horizon_length: {config.horizon_length}")
    return model
