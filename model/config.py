"""
Config classes for model
"""

from dataclasses import dataclass, field
from typing import Literal

# Model architecture config
@dataclass
class ModelConfig:
    input_patch_len: int = 8          
    output_patch_len: int = 32        
    context_patches: int = 4          
    horizon_patches: int = 1          
    
    # Transformer dims
    d_model: int = 128                
    n_heads: int = 4                  
    n_layers: int = 4                 
    d_ff: int = 512                   
    dropout: float = 0.1              
    
    # Input features
    n_price_features: int = 1         
    n_aux_features: int = 1           
    n_categories: int = 50            
    category_embed_dim: int = 16      
    
    # Output settings
    use_sigmoid: bool = True          
    
    # Total context length in time steps
    @property
    def context_length(self) -> int:
        return self.input_patch_len * self.context_patches
    
    # Total horizon length in time steps  
    @property
    def horizon_length(self) -> int:
        return self.output_patch_len * self.horizon_patches
    
    # Ratio of output patch length to input patch length
    @property
    def output_input_ratio(self) -> int:
        return self.output_patch_len // self.input_patch_len


# Training config
@dataclass
class TrainingConfig:
    # Data
    data_path: str = "polymarket_data.parquet"
    train_split: float = 0.8          
    val_split: float = 0.1            
    
    # Training
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 50
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    
    # Scheduler
    scheduler: Literal["cosine", "plateau", "none"] = "cosine"
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 5               
    
    # Logging
    log_every: int = 100              
    
    # Hardware
    device: str = "cuda"              
    mixed_precision: bool = True      
    num_workers: int = 4              
    
    # Reproducibility
    seed: int = 42


# Data processing config
@dataclass  
class DataConfig:
    # Sequence settings
    context_length: int = 32          
    horizon_length: int = 32          
    stride: int = 8                   
    
    # Normalization
    normalize_prices: bool = True
    normalize_hours: bool = True
    
    # Category handling
    min_category_count: int = 10      
    unknown_category: str = "Unknown"
    
    # Filtering
    min_sequence_length: int = 64     
    max_markets: int | None = None    

