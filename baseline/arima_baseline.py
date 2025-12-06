"""
ARIMA (autogressive integrated moving average) baseline
Note: ARIMA is univariate and does not use hours_to_resolution
"""

import argparse
import pickle
import warnings
from pathlib import Path
from typing import Tuple, Optional
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

DEFAULT_ORDER = (5, 1, 2)  # (p, d, q)
CONTEXT_LENGTH = 256
HORIZON_LENGTH = 128
STRIDE = 32
SEED = 42
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1

def create_sequences(
    df: pd.DataFrame,
    context_length: int = CONTEXT_LENGTH,
    horizon_length: int = HORIZON_LENGTH,
    stride: int = STRIDE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


# split data by market for zero-shot evals
def split_by_market(
    X: np.ndarray,
    y: np.ndarray,
    market_ids: np.ndarray,
    train_split: float = TRAIN_SPLIT,
    val_split: float = VAL_SPLIT,
    seed: int = SEED,
    processor_path: str = None,
) -> dict:
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


# Augmented Dickey-Fuller test
def check_stationarity(series: np.ndarray, significance: float = 0.05) -> Tuple[bool, float]:
    try:
        result = adfuller(series, autolag='AIC')
        p_value = result[1]
        is_stationary = p_value < significance
        return is_stationary, p_value
    except Exception:
        return False, 1.0


# fit ARIMA model and forecast horizon steps ahead
def fit_arima_and_forecast(
    context: np.ndarray,
    horizon: int,
    order: Tuple[int, int, int] = DEFAULT_ORDER,
    max_retries: int = 3,
) -> np.ndarray:
    p, d, q = order
    
    orders_to_try = [
        (p, d, q),
        (p, d, 0),
        (2, d, 0),
        (1, d, 0),
        (0, d, 0),
    ]
    
    for attempt_order in orders_to_try[:max_retries + 1]:
        try:
            model = ARIMA(context, order=attempt_order)
            fitted = model.fit()
            forecast = fitted.forecast(steps=horizon)
            forecast = np.clip(forecast, 0.0, 1.0)
            return forecast
        except Exception:
            continue
    
    # fallback to naive forecast
    return np.full(horizon, context[-1])

# fit ARIMA and generate forecasts for a batch of sequences
def arima_forecast_batch(
    contexts: np.ndarray,
    horizon: int,
    order: Tuple[int, int, int] = DEFAULT_ORDER,
    verbose: bool = True,
    max_samples: Optional[int] = None,
) -> np.ndarray:
    n_samples = len(contexts)
    
    # apply sample cap if specified
    if max_samples is not None and n_samples > max_samples:
        print(f"Limiting to {max_samples:,} samples (from {n_samples:,})")
        indices = np.random.choice(n_samples, max_samples, replace=False)
        contexts = contexts[indices]
        n_samples = max_samples
    else:
        indices = None
    
    predictions = np.zeros((n_samples, horizon))
    start_time = time.time()
    
    for i in range(n_samples):
        predictions[i] = fit_arima_and_forecast(contexts[i], horizon, order)
        
        if verbose and (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (n_samples - i - 1) / rate
            print(f"  Progress: {i+1:,}/{n_samples:,} ({100*(i+1)/n_samples:.1f}%) "
                  f"| {rate:.1f} seq/s | ETA: {remaining:.0f}s")
    
    return predictions, indices


# MAE, MSE, RMSE
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse}


# ARIMA eval
def evaluate_arima_baseline(
    data_path: str,
    order: Tuple[int, int, int] = DEFAULT_ORDER,
    context_length: int = CONTEXT_LENGTH,
    horizon_length: int = HORIZON_LENGTH,
    stride: int = STRIDE,
    train_split: float = TRAIN_SPLIT,
    val_split: float = VAL_SPLIT,
    seed: int = SEED,
    processor_path: str = None,
    max_test_samples: Optional[int] = None,
) -> dict:
    print(f"ARIMA Order: (p={order[0]}, d={order[1]}, q={order[2]})")
    print(f"AR (p={order[0]}): Autoregressive terms")
    print(f"I (d={order[1]}): Differencing order")
    print(f"MA (q={order[2]}): Moving average terms")
    print(f"Context length: {context_length}")
    print(f"Horizon length: {horizon_length}")
    print(f"Stride: {stride}")
    print(f"Seed: {seed}")
    
    print(f"\nLoading data from {data_path}")
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df):,} rows")
    
    print("\nCreating sequences")
    X, y, market_ids = create_sequences(df, context_length, horizon_length, stride)
    print(f"Total sequences: {len(X):,}")
    print(f"Context shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Split by market
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
    
    # check stationarity
    print("\nChecking stationarity on sample of training data")
    sample_indices = np.random.choice(len(X_train), min(100, len(X_train)), replace=False)
    stationary_count = sum(check_stationarity(X_train[idx])[0] for idx in sample_indices)
    print(f"Stationary series: {stationary_count}/{len(sample_indices)} "
          f"({100*stationary_count/len(sample_indices):.1f}%)")
    
    print(f"\nFitting ARIMA models and generating forecasts on test set")
    
    start_time = time.time()
    y_pred_test, sample_indices = arima_forecast_batch(
        X_test, horizon_length, order,
        verbose=True,
        max_samples=max_test_samples,
    )
    elapsed = time.time() - start_time
    
    # if we sampled, also sample targets
    if sample_indices is not None:
        y_test_eval = y_test[sample_indices]
    else:
        y_test_eval = y_test
    
    print(f"Completed in {elapsed:.1f}s ({len(y_pred_test)/elapsed:.1f} sequences/second)")
    
    # metrics
    test_metrics = compute_metrics(y_test_eval.flatten(), y_pred_test.flatten())
    
    # naive baseline for scaled MAE
    naive_preds = np.repeat(X_test[:len(y_pred_test), -1:], horizon_length, axis=1)
    if sample_indices is not None:
        naive_preds = naive_preds[sample_indices]
    naive_metrics = compute_metrics(y_test_eval.flatten(), naive_preds.flatten())
    
    scaled_mae = test_metrics['MAE'] / naive_metrics['MAE']
    
    print("RESULTS")
    
    print("\nTest Set (Zero-Shot on Held-Out Markets):")
    print(f"\nMAE: {test_metrics['MAE']:.6f}")
    print(f"MSE: {test_metrics['MSE']:.6f}")
    print(f"RMSE: {test_metrics['RMSE']:.6f}")
    
    print("SCALED MAE:")
    print(f"Naive Baseline MAE: {naive_metrics['MAE']:.6f}")
    print(f"ARIMA MAE: {test_metrics['MAE']:.6f}")
    print(f"Scaled MAE: {scaled_mae:.4f}")
    if scaled_mae < 1.0:
        print(f" so ARIMA beats naive baseline by {(1-scaled_mae)*100:.1f}%")
    else:
        print(f" so naive baseline beats ARIMA by {(scaled_mae-1)*100:.1f}%")
    
    # metrics by horizon
    print("ARIMA MAE BY FORECAST HORIZON")
    
    horizons = [10, 32, 64, 128]
    horizon_metrics = {}
    for h in horizons:
        if h <= horizon_length:
            y_pred_h = y_pred_test[:, :h]
            y_true_h = y_test_eval[:, :h]
            mae_h = np.mean(np.abs(y_true_h - y_pred_h))
            horizon_metrics[h] = mae_h
            print(f"Horizon {h:3d} steps: MAE = {mae_h:.6f}")
    
    print(f"\nARIMA Baseline MAE:  {test_metrics['MAE']:.6f}")
    print(f"ARIMA Baseline RMSE: {test_metrics['RMSE']:.6f}")
    print(f"ARIMA Scaled MAE:    {scaled_mae:.4f}")
    
    return {
        'test_mae': test_metrics['MAE'],
        'test_mse': test_metrics['MSE'],
        'test_rmse': test_metrics['RMSE'],
        'naive_mae': naive_metrics['MAE'],
        'scaled_mae': scaled_mae,
        'horizon_metrics': horizon_metrics,
        'order': order,
    }


# simple grid search for ARIMA order
def grid_search_order(
    data_path: str,
    context_length: int = CONTEXT_LENGTH,
    horizon_length: int = HORIZON_LENGTH,
    max_samples: int = 500,
    seed: int = SEED,
) -> Tuple[int, int, int]:
    print("ARIMA ORDER GRID SEARCH")
    
    orders_to_try = [
        (1, 1, 0),
        (2, 1, 0),
        (1, 1, 1),
        (2, 1, 1),
        (3, 1, 1),
        (5, 1, 2),
    ]
    
    df = pd.read_parquet(data_path)
    X, y, market_ids = create_sequences(df, context_length, horizon_length)
    
    # with cap
    if len(X) > max_samples:
        indices = np.random.RandomState(seed).choice(len(X), max_samples, replace=False)
        X = X[indices]
        y = y[indices]
    
    print(f"Testing on {len(X)} samples...")
    
    best_order = orders_to_try[0]
    best_mae = float('inf')
    
    for order in orders_to_try:
        print(f"\nTesting order {order}...")
        try:
            y_pred, _ = arima_forecast_batch(X, horizon_length, order, verbose=False)
            mae = np.mean(np.abs(y - y_pred))
            print(f"  MAE: {mae:.6f}")
            
            if mae < best_mae:
                best_mae = mae
                best_order = order
        except Exception as e:
            print(f"  Failed: {e}")
    
    print(f"\nBest order: {best_order} with MAE: {best_mae:.6f}")
    return best_order


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--order", type=str, default="5,1,2")
    
    # sequence params
    parser.add_argument("--context-length", type=int, default=CONTEXT_LENGTH)
    parser.add_argument("--horizon-length", type=int, default=HORIZON_LENGTH)
    parser.add_argument("--stride", type=int, default=STRIDE)
    
    # split params
    parser.add_argument("--train-split", type=float, default=TRAIN_SPLIT)
    parser.add_argument("--val-split", type=float, default=VAL_SPLIT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--processor", type=str, default=None)
    
    # perf options
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--max-grid-samples", type=int, default=500)
    parser.add_argument("--grid-search", action="store_true")
    
    args = parser.parse_args()
    
    order = tuple(int(x) for x in args.order.split(","))
    if len(order) != 3:
        raise ValueError("Order must be 'p,d,q' (e.g., '5,1,2')")
    
    if args.grid_search:
        best_order = grid_search_order(
            data_path=args.data,
            context_length=args.context_length,
            horizon_length=args.horizon_length,
            max_samples=args.max_grid_samples,
            seed=args.seed,
        )
        order = best_order
    
    evaluate_arima_baseline(
        data_path=args.data,
        order=order,
        context_length=args.context_length,
        horizon_length=args.horizon_length,
        stride=args.stride,
        train_split=args.train_split,
        val_split=args.val_split,
        seed=args.seed,
        processor_path=args.processor,
        max_test_samples=args.max_test_samples,
    )


if __name__ == "__main__":
    main()
