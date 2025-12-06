"""
Evaluation script for zero-shot forecasting metrics: 
Computing MAE, MSE, RMSE (overall and per-category), comparison with baseline metrics, and per-market analysis  
"""
import sys
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.config import ModelConfig, DataConfig
from model.model import PredictionMarketTimesFM
from model.dataset import DataProcessor, PredictionMarketDataset


# Load model from checkpoint
def load_model(checkpoint_path: str, processor: DataProcessor = None, device: str = "cuda") -> tuple:
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load model config from processor
    if processor is not None and hasattr(processor, 'model_config'):
        model_config = processor.model_config
        print(f"Loaded model config from processor")
    else:
        # Fallback to default config with warning
        model_config = ModelConfig()
        print("WARNING: Using default ModelConfig. This may not match trained model!")
    
    model = PredictionMarketTimesFM(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"  d_model: {model_config.d_model}, n_layers: {model_config.n_layers}")
    print(f"  n_categories: {model_config.n_categories}")
    
    return model, model_config, device


# Evaluate model on test set
@torch.no_grad()
def evaluate_model(
    model: PredictionMarketTimesFM,
    test_loader: DataLoader,
    processor: DataProcessor,
    device: torch.device,
) -> dict:
    model.eval()
    
    all_preds = []
    all_targets = []
    all_market_ids = []
    all_category_ids = []
    
    for batch_idx, batch in enumerate(test_loader):
        context_patches = batch['context_patches'].to(device)
        target_patches = batch['target_patches'].to(device)
        category_ids = batch['category_id'].to(device)
        
        # Create mask
        batch_size, num_patches, patch_len, _ = context_patches.shape
        mask = torch.zeros(batch_size, num_patches, patch_len, 
                         dtype=torch.bool, device=device)
        
        # Forward pass
        predictions, _, _ = model(context_patches, mask, category_ids)
        
        # Get horizon predictions
        horizon_patches = target_patches.size(1)
        pred_horizon = predictions[:, -horizon_patches:, :]
        
        all_preds.append(pred_horizon.cpu().numpy())
        all_targets.append(target_patches.cpu().numpy())
        all_category_ids.append(category_ids.cpu().numpy())
        
        if batch_idx % 100 == 0:
            print(f"Processed batch {batch_idx}/{len(test_loader)}")
    
    # Concatenate
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_category_ids = np.concatenate(all_category_ids, axis=0)
    
    # Flatten for overall metrics
    preds_flat = all_preds.flatten()
    targets_flat = all_targets.flatten()
    
    # Inverse transform to original scale
    preds_orig = processor.inverse_transform_prices(preds_flat)
    targets_orig = processor.inverse_transform_prices(targets_flat)
    
    # Overall metrics
    mae = np.mean(np.abs(preds_orig - targets_orig))
    mse = np.mean((preds_orig - targets_orig) ** 2)
    rmse = np.sqrt(mse)
    
    # Per-category metrics
    category_metrics = {}
    if processor.category_encoder is not None:
        categories = processor.category_encoder.classes_
        
        # Reshape for per-sample metrics
        n_samples = all_preds.shape[0]
        preds_per_sample = all_preds.reshape(n_samples, -1)
        targets_per_sample = all_targets.reshape(n_samples, -1)
        
        for cat_idx, cat_name in enumerate(categories):
            mask = all_category_ids == cat_idx
            if mask.sum() == 0:
                continue
            
            cat_preds = processor.inverse_transform_prices(preds_per_sample[mask].flatten())
            cat_targets = processor.inverse_transform_prices(targets_per_sample[mask].flatten())
            
            category_metrics[cat_name] = {
                'mae': np.mean(np.abs(cat_preds - cat_targets)),
                'mse': np.mean((cat_preds - cat_targets) ** 2),
                'rmse': np.sqrt(np.mean((cat_preds - cat_targets) ** 2)),
                'n_samples': mask.sum(),
            }
    
    return {
        'overall': {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'n_samples': len(preds_flat),
        },
        'per_category': category_metrics,
        'predictions': preds_orig,
        'targets': targets_orig,
    }


# Print evaluation results
def print_results(results: dict, baseline_metrics: dict = None):
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    overall = results['overall']
    print(f"\nOverall Metrics ({overall['n_samples']:,} predictions):")
    print(f"  MAE:  {overall['mae']:.6f}")
    print(f"  MSE:  {overall['mse']:.6f}")
    print(f"  RMSE: {overall['rmse']:.6f}")
    
    if baseline_metrics:
        print(f"\nComparison with LSTM Baseline:")
        print(f"  Baseline MAE:  {baseline_metrics['mae']:.6f}")
        print(f"  Baseline MSE:  {baseline_metrics['mse']:.6f}")
        print(f"  Baseline RMSE: {baseline_metrics['rmse']:.6f}")
        
        mae_improvement = (baseline_metrics['mae'] - overall['mae']) / baseline_metrics['mae'] * 100
        rmse_improvement = (baseline_metrics['rmse'] - overall['rmse']) / baseline_metrics['rmse'] * 100
        
        print(f"\n  MAE Improvement:  {mae_improvement:+.2f}%")
        print(f"  RMSE Improvement: {rmse_improvement:+.2f}%")
    
    if results['per_category']:
        print(f"\nPer-Category Metrics (top 10 by sample count):")
        sorted_cats = sorted(
            results['per_category'].items(),
            key=lambda x: x[1]['n_samples'],
            reverse=True
        )[:10]
        
        print(f"  {'Category':<25} {'MAE':>10} {'RMSE':>10} {'Samples':>10}")
        print("  " + "-"*55)
        for cat_name, metrics in sorted_cats:
            print(f"  {cat_name:<25} {metrics['mae']:>10.6f} {metrics['rmse']:>10.6f} {metrics['n_samples']:>10,}")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate TimesFM-inspired model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--processor", type=str, required=True, help="Path to processor.pkl")
    parser.add_argument("--data", type=str, required=True, help="Path to parquet data")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--output", type=str, default=None, help="Output CSV for predictions")
    
    args = parser.parse_args()
    
    # Load processor
    processor = DataProcessor.load(args.processor)
    
    # Load model using config from processor
    model, model_config, device = load_model(args.checkpoint, processor, args.device)
    
    # Load test data
    print(f"\nLoading data from {args.data}...")
    df = pd.read_parquet(args.data)
    
    # Filter to test markets only
    test_df = df[df['market_id'].isin(processor.test_markets)]
    print(f"Test markets: {len(processor.test_markets)}")
    print(f"Test rows: {len(test_df):,}")
    
    # Create test sequences
    data_config = DataConfig(
        context_length=model_config.context_length,
        horizon_length=model_config.horizon_length,
        stride=model_config.input_patch_len,
    )
    
    # Process category encoding
    test_df = test_df.copy()
    test_df['category'] = test_df['category'].fillna('Unknown')
    test_df['category_id'] = processor.category_encoder.transform(
        test_df['category'].apply(
            lambda x: x if x in processor.category_encoder.classes_ else 'Unknown'
        )
    )
    
    # Create sequences
    test_data = processor._create_sequences(test_df)
    test_data = processor._transform(test_data)
    
    # Create dataset
    test_dataset = PredictionMarketDataset(
        sequences=test_data['sequences'],
        targets=test_data['targets'],
        category_ids=test_data['category_ids'],
        market_ids=test_data['market_ids'],
        config=model_config,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    print(f"Test sequences: {len(test_dataset):,}")
    
    print("\nRunning evaluation...")
    results = evaluate_model(model, test_loader, processor, device)
    
    # Baseline metrics from the notebook
    baseline_metrics = {
        'mae': 0.002265,
        'mse': 0.000132,
        'rmse': 0.011495,
    }
    
    print_results(results, baseline_metrics)
    
    # Save predictions
    if args.output:
        pred_df = pd.DataFrame({
            'prediction': results['predictions'],
            'target': results['targets'],
            'error': np.abs(results['predictions'] - results['targets']),
        })
        pred_df.to_csv(args.output, index=False)
        print(f"\nSaved predictions to {args.output}")


if __name__ == "__main__":
    main()

