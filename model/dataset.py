"""
Dataset and data loading utilities for prediction market data:
Loading Parquet files, windowing sequences into context/horizon patches, train/val/test splits by market (for zero-shot evaluation), normalization and category encoding  
"""

import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pickle

from model.config import ModelConfig, DataConfig

# Set up logger for this module
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Dataset for prediction market time series
class PredictionMarketDataset(Dataset):
    def __init__(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        category_ids: np.ndarray,
        market_ids: np.ndarray,
        config: ModelConfig,
    ):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        self.category_ids = torch.LongTensor(category_ids)
        self.market_ids = market_ids
        self.config = config
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]   # (context_length, n_features)
        target = self.targets[idx]  # (horizon_length,)
        cat_id = self.category_ids[idx]
        
        # (context_length, n_features) -> (num_patches, patch_len, n_features)
        context_patches = self._patchify(seq, self.config.input_patch_len)
        
        # (horizon_length,) -> (horizon_patches, output_patch_len)
        target_patches = self._patchify_1d(target, self.config.output_patch_len)
        
        return {
            "context_patches": context_patches,
            "target_patches": target_patches,
            "category_id": cat_id,
        }
    
    # Convert sequence to patches
    def _patchify(self, seq: torch.Tensor, patch_len: int) -> torch.Tensor:
        seq_len, n_features = seq.shape
        num_patches = seq_len // patch_len
        seq = seq[:num_patches * patch_len]
        return seq.view(num_patches, patch_len, n_features)
    
    # Convert 1D sequence to patches
    def _patchify_1d(self, seq: torch.Tensor, patch_len: int) -> torch.Tensor:
        seq_len = seq.shape[0]
        num_patches = seq_len // patch_len
        seq = seq[:num_patches * patch_len]
        return seq.view(num_patches, patch_len)


# Process raw Parquet data into training-ready datasets
class DataProcessor:
    # Initialize processor
    def __init__(
        self,
        model_config: ModelConfig,
        data_config: DataConfig,
    ):
        self.model_config = model_config
        self.data_config = data_config
        
        # Will be fitted
        self.price_scaler: Optional[StandardScaler] = None
        self.hours_scaler: Optional[StandardScaler] = None
        self.category_encoder: Optional[LabelEncoder] = None
        
        # Metadata
        self.train_markets: List[str] = []
        self.val_markets: List[str] = []
        self.test_markets: List[str] = []
    
    # Load data and create train/val/test datasets
    def load_and_process(
        self,
        data_path: str | Path,
        train_split: float = 0.8,
        val_split: float = 0.1,
        seed: int = 42,
    ) -> Tuple[PredictionMarketDataset, PredictionMarketDataset, PredictionMarketDataset]:
        # Load data
        import time
        print(f"Loading data from {data_path}...")
        t0 = time.time()
        df = pd.read_parquet(data_path)
        print(f"Loaded {len(df):,} rows in {time.time()-t0:.1f}s")
        
        # Sort by market, token, timestamp
        print("Sorting data...")
        t0 = time.time()
        df = df.sort_values(['market_id', 'token_id', 'timestamp'])
        print(f"Sorted in {time.time()-t0:.1f}s")
        
        # Filter columns
        required_cols = ['price', 'hours_to_resolution', 'market_id', 'token_id', 'category', 'timestamp']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Remove NaN/Inf values 
        print("Cleaning data (removing NaN/Inf values)...")
        initial_rows = len(df)
        
        nan_mask = df[['price', 'hours_to_resolution']].isna().any(axis=1)
        inf_mask = df['price'].isin([np.inf, -np.inf]) | df['hours_to_resolution'].isin([np.inf, -np.inf])
        
        # Remove bad rows
        bad_mask = nan_mask | inf_mask
        if bad_mask.sum() > 0:
            print(f"  Removing {bad_mask.sum():,} rows with NaN/Inf values")
            df = df[~bad_mask].copy()
        
        # Clip prices to [0, 1] range
        price_below_0 = (df['price'] < 0).sum()
        price_above_1 = (df['price'] > 1).sum()
        if price_below_0 > 0 or price_above_1 > 0:
            print(f"  Clipping {price_below_0:,} prices below 0 and {price_above_1:,} prices above 1")
            df['price'] = df['price'].clip(0.0, 1.0)
        
        # Clip hours_to_resolution to non-negative
        negative_hours = (df['hours_to_resolution'] < 0).sum()
        if negative_hours > 0:
            print(f"  Clipping {negative_hours:,} negative hours_to_resolution values to 0")
            df['hours_to_resolution'] = df['hours_to_resolution'].clip(lower=0.0)
        
        print(f"  Cleaned: {initial_rows:,} -> {len(df):,} rows ({initial_rows - len(df):,} removed)")
        
        # Handle category encoding
        df['category'] = df['category'].fillna(self.data_config.unknown_category)
        self.category_encoder = LabelEncoder()
        df['category_id'] = self.category_encoder.fit_transform(df['category'])
        
        # Update model config with actual number of categories
        n_categories = len(self.category_encoder.classes_)
        self.model_config.n_categories = n_categories
        print(f"Found {n_categories} unique categories (updated model_config.n_categories)")
        
        # Split markets into train/val/test
        # train_split = fraction for train+val combined (e.g., 0.8 means 80% train+val, 20% test)
        # val_split = fraction of total for validation (e.g., 0.1 means 10% val)
        unique_markets = df['market_id'].unique()
        np.random.seed(seed)
        np.random.shuffle(unique_markets)
        
        n_total = len(unique_markets)
        n_train_val = int(n_total * train_split)  
        n_val = int(n_total * val_split)          
        n_train_final = n_train_val - n_val       
        
        self.train_markets = list(unique_markets[:n_train_final])
        self.val_markets = list(unique_markets[n_train_final:n_train_val])
        self.test_markets = list(unique_markets[n_train_val:])
        
        print(f"Markets split: {len(self.train_markets)} train, {len(self.val_markets)} val, {len(self.test_markets)} test")
        
        # Create sequences for each split
        print("\nCreating training sequences...")
        t0 = time.time()
        train_data = self._create_sequences(df[df['market_id'].isin(self.train_markets)])
        print(f"Training sequences created in {time.time()-t0:.1f}s")
        
        print("\nCreating validation sequences...")
        t0 = time.time()
        val_data = self._create_sequences(df[df['market_id'].isin(self.val_markets)])
        print(f"Validation sequences created in {time.time()-t0:.1f}s")
        
        print("\nCreating test sequences...")
        t0 = time.time()
        test_data = self._create_sequences(df[df['market_id'].isin(self.test_markets)])
        print(f"Test sequences created in {time.time()-t0:.1f}s")
        
        # Fit scalers on training data only
        self._fit_scalers(train_data)
        
        # Transform all splits
        train_data = self._transform(train_data)
        val_data = self._transform(val_data)
        test_data = self._transform(test_data)
        
        # Create datasets
        train_dataset = PredictionMarketDataset(
            sequences=train_data['sequences'],
            targets=train_data['targets'],
            category_ids=train_data['category_ids'],
            market_ids=train_data['market_ids'],
            config=self.model_config,
        )
        
        val_dataset = PredictionMarketDataset(
            sequences=val_data['sequences'],
            targets=val_data['targets'],
            category_ids=val_data['category_ids'],
            market_ids=val_data['market_ids'],
            config=self.model_config,
        )
        
        test_dataset = PredictionMarketDataset(
            sequences=test_data['sequences'],
            targets=test_data['targets'],
            category_ids=test_data['category_ids'],
            market_ids=test_data['market_ids'],
            config=self.model_config,
        )
        
        print(f"Dataset sizes: {len(train_dataset):,} train, {len(val_dataset):,} val, {len(test_dataset):,} test")
        
        return train_dataset, val_dataset, test_dataset
    
    # Create windowed sequences from dataframe
    def _create_sequences(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        context_len = self.data_config.context_length
        horizon_len = self.data_config.horizon_length
        stride = self.data_config.stride
        min_len = self.data_config.min_sequence_length
        
        sequences = []
        targets = []
        category_ids = []
        market_ids = []
        
        groups = list(df.groupby(['market_id', 'token_id']))
        total_groups = len(groups)
        logger.info(f"Processing {total_groups:,} market/token groups...")
        
        for idx, ((market_id, token_id), group) in enumerate(groups):
            if idx % 500 == 0:
                logger.info(f"  Progress: {idx:,}/{total_groups:,} groups ({100*idx/total_groups:.1f}%)")
            
            prices = group['price'].values
            hours = group['hours_to_resolution'].values
            cat_id = group['category_id'].iloc[0]
            
            # Skip short sequences
            if len(prices) < context_len + horizon_len:
                continue
            
            # Window with stride
            for i in range(0, len(prices) - context_len - horizon_len + 1, stride):
                seq_prices = prices[i:i + context_len]
                seq_hours = hours[i:i + context_len]
                target = prices[i + context_len:i + context_len + horizon_len]
                
                # Stack features: [price, hours_to_resolution]
                sequence = np.column_stack([seq_prices, seq_hours])
                
                sequences.append(sequence)
                targets.append(target)
                category_ids.append(cat_id)
                market_ids.append(market_id)
        
        return {
            'sequences': np.array(sequences) if sequences else np.empty((0, context_len, 2)),
            'targets': np.array(targets) if targets else np.empty((0, horizon_len)),
            'category_ids': np.array(category_ids) if category_ids else np.empty((0,), dtype=int),
            'market_ids': np.array(market_ids) if market_ids else np.empty((0,)),
        }
    
    # Fit scalers on training data
    def _fit_scalers(self, data: Dict[str, np.ndarray]) -> None:
        if len(data['sequences']) == 0:
            return
            
        sequences = data['sequences']
        
        if self.data_config.normalize_prices:
            prices = sequences[:, :, 0].flatten().reshape(-1, 1)
            self.price_scaler = StandardScaler()
            self.price_scaler.fit(prices)
            logger.warning("Price normalization is enabled. This may conflict with "
                          "sigmoid output activation for prediction markets.")
        else:
            self.price_scaler = None
        
        if self.data_config.normalize_hours:
            hours = sequences[:, :, 1].flatten().reshape(-1, 1)
            self.hours_scaler = StandardScaler()
            self.hours_scaler.fit(hours)
    
    # Apply fitted scalers to data
    def _transform(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if len(data['sequences']) == 0:
            return data
            
        sequences = data['sequences'].copy()
        targets = data['targets'].copy()
        
        # Normalize prices
        if self.price_scaler is not None:
            n_samples, seq_len, _ = sequences.shape
            prices = sequences[:, :, 0].reshape(-1, 1)
            sequences[:, :, 0] = self.price_scaler.transform(prices).reshape(n_samples, seq_len)
            
            # Also transform targets
            n_samples, target_len = targets.shape
            targets_flat = targets.reshape(-1, 1)
            targets = self.price_scaler.transform(targets_flat).reshape(n_samples, target_len)
        
        # Normalize hours
        if self.hours_scaler is not None:
            n_samples, seq_len, _ = sequences.shape
            hours = sequences[:, :, 1].reshape(-1, 1)
            sequences[:, :, 1] = self.hours_scaler.transform(hours).reshape(n_samples, seq_len)
        
        return {
            'sequences': sequences,
            'targets': targets,
            'category_ids': data['category_ids'],
            'market_ids': data['market_ids'],
        }
    
    # Convert normalized prices back to original scale
    def inverse_transform_prices(self, prices: np.ndarray) -> np.ndarray:
        if self.price_scaler is None:
            return prices
        
        original_shape = prices.shape
        prices_flat = prices.reshape(-1, 1)
        prices_original = self.price_scaler.inverse_transform(prices_flat)
        return prices_original.reshape(original_shape)
    
    # Save processor state (scalers, encoders, market splits)
    def save(self, path: str | Path) -> None:
        state = {
            'price_scaler': self.price_scaler,
            'hours_scaler': self.hours_scaler,
            'category_encoder': self.category_encoder,
            'train_markets': self.train_markets,
            'val_markets': self.val_markets,
            'test_markets': self.test_markets,
            'model_config': self.model_config,
            'data_config': self.data_config,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f"Saved processor to {path}")
    
    # Load processor state
    @classmethod
    def load(cls, path: str | Path) -> "DataProcessor":
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        processor = cls(state['model_config'], state['data_config'])
        processor.price_scaler = state['price_scaler']
        processor.hours_scaler = state['hours_scaler']
        processor.category_encoder = state['category_encoder']
        processor.train_markets = state['train_markets']
        processor.val_markets = state['val_markets']
        processor.test_markets = state['test_markets']
        
        print(f"Loaded processor from {path}")
        return processor


# Create DataLoaders for train/val/test datasets
def create_dataloaders(
    train_dataset: PredictionMarketDataset,
    val_dataset: PredictionMarketDataset,
    test_dataset: PredictionMarketDataset,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader

