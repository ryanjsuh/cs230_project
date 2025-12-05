"""
Compute Scaled MAE, where Scaled MAE = MAE(model) / MAE(naive_last_value)
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.config import ModelConfig
from model.model import PredictionMarketTimesFM
from model.dataset import DataProcessor, PredictionMarketDataset


# Compute MAE for naive last-value baseline
def compute_naive_baseline_mae(test_loader, processor):
    all_naive_preds = []
    all_targets = []
    
    for batch in test_loader:
        context_patches = batch['context_patches']  # (batch, num_patches, patch_len, n_features)
        target_patches = batch['target_patches']    # (batch, horizon_patches, output_patch_len)
        
        # Get last value from context (last time step of last patch, price feature)
        last_values = context_patches[:, -1, -1, 0]  # (batch,)
        
        # Naive prediction: repeat last value for entire horizon
        batch_size = target_patches.size(0)
        horizon_len = target_patches.size(1) * target_patches.size(2)
        naive_pred = last_values.unsqueeze(1).expand(batch_size, horizon_len)
        
        # Flatten targets
        targets_flat = target_patches.view(batch_size, -1)
        
        all_naive_preds.append(naive_pred.numpy())
        all_targets.append(targets_flat.numpy())
    
    all_naive_preds = np.concatenate(all_naive_preds, axis=0).flatten()
    all_targets = np.concatenate(all_targets, axis=0).flatten()
    
    # Inverse transform to original scale
    naive_preds_orig = processor.inverse_transform_prices(all_naive_preds)
    targets_orig = processor.inverse_transform_prices(all_targets)
    
    naive_mae = np.mean(np.abs(naive_preds_orig - targets_orig))
    return naive_mae


# Compute MAE for model predictions
@torch.no_grad()
def compute_model_mae(model, test_loader, processor, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    for batch in test_loader:
        context_patches = batch['context_patches'].to(device)
        target_patches = batch['target_patches']
        category_ids = batch['category_id'].to(device)
        
        # Create mask
        batch_size, num_patches, patch_len, _ = context_patches.shape
        mask = torch.zeros(batch_size, num_patches, patch_len, 
                         dtype=torch.bool, device=device)
        
        # Forward pass
        predictions, _, _ = model(context_patches, mask, category_ids)
        
        # Get horizon predictions
        horizon_patches = target_patches.size(1)
        pred_horizon = predictions[:, -horizon_patches:, :].cpu()
        
        all_preds.append(pred_horizon.numpy())
        all_targets.append(target_patches.numpy())
    
    all_preds = np.concatenate(all_preds, axis=0).flatten()
    all_targets = np.concatenate(all_targets, axis=0).flatten()
    
    # Inverse transform
    preds_orig = processor.inverse_transform_prices(all_preds)
    targets_orig = processor.inverse_transform_prices(all_targets)
    
    model_mae = np.mean(np.abs(preds_orig - targets_orig))
    return model_mae


def main():
    parser = argparse.ArgumentParser(description="Compute Scaled MAE")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--processor", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    
    # Load processor
    processor = DataProcessor.load(args.processor)
    model_config = processor.model_config
    
    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = PredictionMarketTimesFM(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load test data
    print(f"Loading data from {args.data}...")
    df = pd.read_parquet(args.data)
    test_df = df[df['market_id'].isin(processor.test_markets)].copy()
    
    # Process
    test_df['category'] = test_df['category'].fillna('Unknown')
    test_df['category_id'] = processor.category_encoder.transform(
        test_df['category'].apply(
            lambda x: x if x in processor.category_encoder.classes_ else 'Unknown'
        )
    )
    
    from model.config import DataConfig
    data_config = DataConfig(
        context_length=model_config.context_length,
        horizon_length=model_config.horizon_length,
        stride=model_config.input_patch_len,
    )
    processor.data_config = data_config
    
    test_data = processor._create_sequences(test_df)
    test_data = processor._transform(test_data)
    
    test_dataset = PredictionMarketDataset(
        sequences=test_data['sequences'],
        targets=test_data['targets'],
        category_ids=test_data['category_ids'],
        market_ids=test_data['market_ids'],
        config=model_config,
    )
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Test samples: {len(test_dataset):,}")
    
    # Compute metrics
    print("\nComputing naive baseline MAE...")
    naive_mae = compute_naive_baseline_mae(test_loader, processor)
    print(f"Naive Baseline MAE: {naive_mae:.6f}")
    
    print("\nComputing model MAE...")
    model_mae = compute_model_mae(model, test_loader, processor, device)
    print(f"Model MAE: {model_mae:.6f}")
    
    # Scaled MAE
    scaled_mae = model_mae / naive_mae
    
    print("\n" + "="*60)
    print("SCALED MAE RESULTS (TimesFM comparison)")
    print("="*60)
    print(f"Naive Last-Value MAE:  {naive_mae:.6f}")
    print(f"Model MAE:             {model_mae:.6f}")
    print(f"Scaled MAE:            {scaled_mae:.4f}")
    print("="*60)
    
    if scaled_mae < 1.0:
        print(f"✓ Model beats naive baseline by {(1-scaled_mae)*100:.1f}%")
    else:
        print(f"✗ Model underperforms naive baseline by {(scaled_mae-1)*100:.1f}%")
    
    print("\nFor reference:")
    print("  TimesFM (zero-shot): ~0.85-0.95 scaled MAE on Monash benchmarks")
    print("  < 1.0 = better than naive, < 0.9 = good, < 0.8 = excellent")


if __name__ == "__main__":
    main()

