"""
Naive last-value baseline used by TimesFM for computing scaled MAE:
For each series, predict the last observed value for all future timesteps
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Predict last context value for entire horizon
def naive_last_value_forecast(context: np.ndarray, horizon: int) -> np.ndarray:
    if context.ndim == 2:
        # Univariate: (B, L) -> (B, H)
        last_values = context[:, -1:]  # (B, 1)
        preds = np.repeat(last_values, horizon, axis=1)  # (B, H)
    else:
        # Multivariate: (B, L, D) -> (B, H, D)
        last_values = context[:, -1:, :]  # (B, 1, D)
        preds = np.repeat(last_values, horizon, axis=1)  # (B, H, D)
    return preds


# Create windowed sequences from dataframe
def create_sequences(df: pd.DataFrame, context_length: int, horizon_length: int, stride: int = 1):
    sequences = []
    targets = []
    market_ids = []
    
    df = df.sort_values(['market_id', 'token_id', 'timestamp'])
    
    for (market_id, token_id), group in df.groupby(['market_id', 'token_id']):
        prices = group['price'].values
        
        if len(prices) < context_length + horizon_length:
            continue
        
        for i in range(0, len(prices) - context_length - horizon_length + 1, stride):
            context = prices[i:i + context_length]
            target = prices[i + context_length:i + context_length + horizon_length]
            
            sequences.append(context)
            targets.append(target)
            market_ids.append(market_id)
    
    return np.array(sequences), np.array(targets), np.array(market_ids)


# Evaluate naive last-value baseline on prediction market data
def evaluate_naive_baseline(
    data_path: str,
    context_length: int = 256,
    horizon_length: int = 128,
    stride: int = 32,
    test_split: float = 0.2,
    seed: int = 42,
):
    print("="*60)
    print("NAIVE LAST-VALUE BASELINE")
    print("="*60)
    print(f"Context length: {context_length}")
    print(f"Horizon length: {horizon_length}")
    print(f"Stride: {stride}")
    print("="*60)
    
    # Load data
    print(f"\nLoading data from {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df):,} rows")
    
    # Create sequences
    print("\nCreating sequences...")
    X, y, market_ids = create_sequences(df, context_length, horizon_length, stride)
    print(f"Total sequences: {len(X):,}")
    print(f"Context shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Split by market (zero-shot evaluation)
    unique_markets = np.unique(market_ids)
    np.random.seed(seed)
    train_markets, test_markets = train_test_split(
        unique_markets, test_size=test_split, random_state=seed
    )
    
    train_mask = np.isin(market_ids, train_markets)
    test_mask = np.isin(market_ids, test_markets)
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"\nTrain sequences: {len(X_train):,} ({len(train_markets)} markets)")
    print(f"Test sequences: {len(X_test):,} ({len(test_markets)} markets)")
    
    # Generate naive predictions (no training needed!)
    print("\nGenerating naive baseline predictions...")
    y_pred_train = naive_last_value_forecast(X_train, horizon_length)
    y_pred_test = naive_last_value_forecast(X_test, horizon_length)
    
    # Compute metrics
    def compute_metrics(y_true, y_pred):
        mae = np.mean(np.abs(y_true - y_pred))
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        return mae, mse, rmse
    
    train_mae, train_mse, train_rmse = compute_metrics(y_train.flatten(), y_pred_train.flatten())
    test_mae, test_mse, test_rmse = compute_metrics(y_test.flatten(), y_pred_test.flatten())
    
    # Results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    print("\nTraining Set (for reference):")
    print(f"  MAE:  {train_mae:.6f}")
    print(f"  MSE:  {train_mse:.6f}")
    print(f"  RMSE: {train_rmse:.6f}")
    
    print("\nTest Set (Zero-Shot on Held-Out Markets):")
    print(f"  MAE:  {test_mae:.6f}")
    print(f"  MSE:  {test_mse:.6f}")
    print(f"  RMSE: {test_rmse:.6f}")
    
    print("\n" + "="*60)
    print("USE THESE FOR SCALED MAE CALCULATION:")
    print("="*60)
    print(f"Naive Baseline MAE: {test_mae:.6f}")
    print(f"\nScaled MAE = Model_MAE / {test_mae:.6f}")
    print(f"\nExample: If your model has MAE=0.10")
    print(f"         Scaled MAE = 0.10 / {test_mae:.6f} = {0.10/test_mae:.4f}")
    print("="*60)
    
    # Also compute for different horizons for reference
    print("\n" + "="*60)
    print("NAIVE BASELINE MAE BY FORECAST HORIZON")
    print("="*60)
    
    horizons = [10, 32, 64, 128]
    for h in horizons:
        if h <= horizon_length:
            # Use only first h steps of predictions and targets
            y_pred_h = y_pred_test[:, :h]
            y_true_h = y_test[:, :h]
            mae_h = np.mean(np.abs(y_true_h - y_pred_h))
            print(f"  Horizon {h:3d} steps: MAE = {mae_h:.6f}")
    
    return {
        'train_mae': train_mae,
        'train_mse': train_mse,
        'train_rmse': train_rmse,
        'test_mae': test_mae,
        'test_mse': test_mse,
        'test_rmse': test_rmse,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate naive last-value baseline for scaled MAE computation"
    )
    parser.add_argument("--data", type=str, required=True, help="Path to parquet data")
    parser.add_argument("--context-length", type=int, default=256, help="Context length")
    parser.add_argument("--horizon-length", type=int, default=128, help="Horizon length")
    parser.add_argument("--stride", type=int, default=32, help="Stride for windowing")
    parser.add_argument("--test-split", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    evaluate_naive_baseline(
        data_path=args.data,
        context_length=args.context_length,
        horizon_length=args.horizon_length,
        stride=args.stride,
        test_split=args.test_split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

