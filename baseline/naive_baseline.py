"""
Naive last-value baseline for computing scaled MAE:
For each series, predict the last observed value for all future timesteps
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# default hyperparams
CONTEXT_LENGTH = 256
HORIZON_LENGTH = 128
STRIDE = 32
SEED = 42
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1


# predict last context value for entire horizon
def naive_last_value_forecast(context: np.ndarray, horizon: int) -> np.ndarray:
    if context.ndim == 1:
        # (L,) to (H,)
        return np.full(horizon, context[-1])
    elif context.ndim == 2:
        # (B, L) to (B, H)
        last_values = context[:, -1:]
        return np.repeat(last_values, horizon, axis=1)
    else:
        # (B, L, D) to use price feature
        last_values = context[:, -1, 0:1]
        return np.repeat(last_values, horizon, axis=1)

def create_sequences(
    df: pd.DataFrame,
    context_length: int,
    horizon_length: int,
    stride: int = STRIDE,
    use_hours: bool = False,
):
    sequences = []
    targets = []
    market_ids = []
    
    df = df.sort_values(['market_id', 'token_id', 'timestamp'])
    
    for (market_id, token_id), group in df.groupby(['market_id', 'token_id']):
        prices = group['price'].values
        
        if len(prices) < context_length + horizon_length:
            continue
        
        if use_hours and 'hours_to_resolution' in group.columns:
            hours = group['hours_to_resolution'].values
        else:
            hours = None
        
        for i in range(0, len(prices) - context_length - horizon_length + 1, stride):
            if hours is not None:
                context = np.column_stack([
                    prices[i:i + context_length],
                    hours[i:i + context_length]
                ])
            else:
                context = prices[i:i + context_length]
            
            target = prices[i + context_length:i + context_length + horizon_length]
            
            sequences.append(context)
            targets.append(target)
            market_ids.append(market_id)
    
    return np.array(sequences), np.array(targets), np.array(market_ids)

# split data by market for zero-shot evals
def split_by_market(
    X: np.ndarray,
    y: np.ndarray,
    market_ids: np.ndarray,
    train_split: float = TRAIN_SPLIT,
    val_split: float = VAL_SPLIT,
    seed: int = SEED,
    processor_path: str = None,
):
    if processor_path is not None:
        with open(processor_path, 'rb') as f:
            state = pickle.load(f)
        train_markets = set(state['train_markets'])
        val_markets = set(state['val_markets'])
        test_markets = set(state['test_markets'])
        print(f"Loaded market splits from {processor_path}")
    else:
        unique_markets = np.unique(market_ids)
        np.random.seed(seed)
        np.random.shuffle(unique_markets)
        
        n_total = len(unique_markets)
        n_train_val = int(n_total * train_split)
        n_val = int(n_total * val_split)
        n_train_final = n_train_val - n_val
        
        train_markets = set(unique_markets[:n_train_final])
        val_markets = set(unique_markets[n_train_final:n_train_val])
        test_markets = set(unique_markets[n_train_val:])
    
    train_mask = np.array([m in train_markets for m in market_ids])
    val_mask = np.array([m in val_markets for m in market_ids])
    test_mask = np.array([m in test_markets for m in market_ids])
    
    return {
        'train': (X[train_mask], y[train_mask]),
        'val': (X[val_mask], y[val_mask]),
        'test': (X[test_mask], y[test_mask]),
        'n_markets': {
            'train': len(train_markets),
            'val': len(val_markets),
            'test': len(test_markets),
        }
    }


# MAE, MSE, RMSE
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse}


# naive last-value baseline eval
def evaluate_naive_baseline(
    data_path: str,
    context_length: int = CONTEXT_LENGTH,
    horizon_length: int = HORIZON_LENGTH,
    stride: int = STRIDE,
    train_split: float = TRAIN_SPLIT,
    val_split: float = VAL_SPLIT,
    seed: int = SEED,
    processor_path: str = None,
    use_hours: bool = False,
):
    print("NAIVE LAST-VALUE BASELINE")
    print(f"\nContext length: {context_length}")
    print(f"Horizon length: {horizon_length}")
    print(f"Stride: {stride}")
    print(f"Seed: {seed}")
    print(f"Features: price" + (" + hours_to_resolution" if use_hours else " only"))
    
    print(f"\nLoading data from {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df):,} rows")
    
    print("\nCreating sequences...")
    X, y, market_ids = create_sequences(
        df, context_length, horizon_length, stride, use_hours
    )
    print(f"Total sequences: {len(X):,}")
    print(f"Context shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    splits = split_by_market(
        X, y, market_ids,
        train_split=train_split,
        val_split=val_split,
        seed=seed,
        processor_path=processor_path,
    )
    
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']
    
    print(f"\nTrain sequences: {len(X_train):,} ({splits['n_markets']['train']} markets)")
    print(f"Val sequences: {len(X_val):,} ({splits['n_markets']['val']} markets)")
    print(f"Test sequences: {len(X_test):,} ({splits['n_markets']['test']} markets)")
    
    print("\nGenerating naive baseline predictions")
    y_pred_train = naive_last_value_forecast(X_train, horizon_length)
    y_pred_val = naive_last_value_forecast(X_val, horizon_length)
    y_pred_test = naive_last_value_forecast(X_test, horizon_length)
    
    train_metrics = compute_metrics(y_train.flatten(), y_pred_train.flatten())
    val_metrics = compute_metrics(y_val.flatten(), y_pred_val.flatten())
    test_metrics = compute_metrics(y_test.flatten(), y_pred_test.flatten())
    
    print("RESULTS")
    
    print("\nTraining Set:")
    print(f"\nMAE: {train_metrics['MAE']:.6f}")
    print(f"MSE: {train_metrics['MSE']:.6f}")
    print(f"RMSE: {train_metrics['RMSE']:.6f}")
    
    print("\nValidation Set:")
    print(f"\nMAE: {val_metrics['MAE']:.6f}")
    print(f"MSE: {val_metrics['MSE']:.6f}")
    print(f"RMSE: {val_metrics['RMSE']:.6f}")
    
    print("\nTest Set (Zero-Shot on Held-Out Markets):")
    print(f"\nMAE: {test_metrics['MAE']:.6f}")
    print(f"MSE: {test_metrics['MSE']:.6f}")
    print(f"RMSE: {test_metrics['RMSE']:.6f}")
    
    print("USE THESE FOR SCALED MAE CALCULATION")
    print(f"\nNaive Baseline MAE: {test_metrics['MAE']:.6f}")
    print(f"\nScaled MAE = Model_MAE / {test_metrics['MAE']:.6f}")
    
    # metrics by horizon
    print("NAIVE BASELINE MAE BY FORECAST HORIZON")
    
    horizons = [10, 32, 64, 128]
    horizon_metrics = {}
    for h in horizons:
        if h <= horizon_length:
            y_pred_h = y_pred_test[:, :h]
            y_true_h = y_test[:, :h]
            mae_h = np.mean(np.abs(y_true_h - y_pred_h))
            horizon_metrics[h] = mae_h
            print(f"Horizon {h:3d} steps: MAE = {mae_h:.6f}")
    
    return {
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics,
        'horizon_metrics': horizon_metrics,
    }


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=str, required=True)
    
    # sequence params
    parser.add_argument("--context-length", type=int, default=CONTEXT_LENGTH)
    parser.add_argument("--horizon-length", type=int, default=HORIZON_LENGTH)
    parser.add_argument("--stride", type=int, default=STRIDE)
    
    # split params
    parser.add_argument("--train-split", type=float, default=TRAIN_SPLIT)
    parser.add_argument("--val-split", type=float, default=VAL_SPLIT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--processor", type=str, default=None)
    
    # feature options
    parser.add_argument("--use-hours", action="store_true", default=False)
    
    args = parser.parse_args()
    
    evaluate_naive_baseline(
        data_path=args.data,
        context_length=args.context_length,
        horizon_length=args.horizon_length,
        stride=args.stride,
        train_split=args.train_split,
        val_split=args.val_split,
        seed=args.seed,
        processor_path=args.processor,
        use_hours=args.use_hours,
    )


if __name__ == "__main__":
    main()
