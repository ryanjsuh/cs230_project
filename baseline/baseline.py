#!/usr/bin/env python3
"""
LSTM baseline
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# default hyperparams
CONTEXT_LENGTH = 256
HORIZON_LENGTH = 128
STRIDE = 32
HIDDEN_SIZE = 64
NUM_LAYERS = 2
SEED = 42
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1


class PriceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int = 64,
        num_layers: int = 2,
        forecast_horizon: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, forecast_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        output = self.fc(last_hidden)
        return output


def load_data(data_path: str) -> pd.DataFrame:
    print(f"Loading data from: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"Dataset shape: {df.shape}")
    return df


# create windowed sequences with stride
def create_sequences(
    df: pd.DataFrame,
    context_length: int = CONTEXT_LENGTH,
    horizon_length: int = HORIZON_LENGTH,
    stride: int = STRIDE,
    use_hours: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = df.sort_values(['market_id', 'token_id', 'timestamp'])
    
    sequences = []
    targets = []
    market_ids = []
    
    for (market_id, token_id), group in df.groupby(['market_id', 'token_id']):
        prices = group['price'].values
        
        if len(prices) < context_length + horizon_length:
            continue
        
        if use_hours and 'hours_to_resolution' in group.columns:
            hours_to_res = group['hours_to_resolution'].values
        else:
            hours_to_res = None
        
        # sliding windows with stride
        for i in range(0, len(prices) - context_length - horizon_length + 1, stride):
            seq_prices = prices[i:i + context_length]
            target = prices[i + context_length:i + context_length + horizon_length]
            
            if hours_to_res is not None:
                seq_hours = hours_to_res[i:i + context_length]
                sequence = np.column_stack([seq_prices, seq_hours])
            else:
                sequence = seq_prices.reshape(-1, 1)
            
            sequences.append(sequence)
            targets.append(target)
            market_ids.append(market_id)
    
    X = np.array(sequences)
    y = np.array(targets)
    market_ids = np.array(market_ids)
    
    print(f"Total sequences: {len(X):,}")
    print(f"Sequence shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    return X, y, market_ids


# split data by market for zero-shot evaluation
def split_by_market(
    X: np.ndarray,
    y: np.ndarray,
    market_ids: np.ndarray,
    train_split: float = TRAIN_SPLIT,
    val_split: float = VAL_SPLIT,
    seed: int = SEED,
    processor_path: str = None,
) -> tuple:
    if processor_path is not None:
        # load splits from saved processor for exact alignment with model
        import pickle
        with open(processor_path, 'rb') as f:
            state = pickle.load(f)
        train_markets = set(state['train_markets'])
        val_markets = set(state['val_markets'])
        test_markets = set(state['test_markets'])
        print(f"Loaded market splits from {processor_path}")
    else:
        # create splits that match model's logic
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
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"Training sequences: {len(X_train):,} ({train_mask.sum()} from {len(train_markets)} markets)")
    print(f"Validation sequences: {len(X_val):,} ({val_mask.sum()} from {len(val_markets)} markets)")
    print(f"Test sequences: {len(X_test):,} ({test_mask.sum()} from {len(test_markets)} markets)")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


# scale data (by default, prices aren't normalized)
def scale_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    normalize_prices: bool = False,
) -> tuple:
    n_train, context_len, n_features = X_train.shape
    n_val = len(X_val)
    n_test = len(X_test)
    
    if normalize_prices:
        # normalize all features
        X_train_flat = X_train.reshape(-1, n_features)
        X_val_flat = X_val.reshape(-1, n_features)
        X_test_flat = X_test.reshape(-1, n_features)
        
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train_flat).reshape(n_train, context_len, n_features)
        X_val_scaled = scaler_X.transform(X_val_flat).reshape(n_val, context_len, n_features)
        X_test_scaled = scaler_X.transform(X_test_flat).reshape(n_test, context_len, n_features)
        
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_val_scaled = scaler_y.transform(y_val)
        y_test_scaled = scaler_y.transform(y_test)
    else:
        # only normalize hours_to_resolution and keep prices as they are
        X_train_scaled = X_train.copy()
        X_val_scaled = X_val.copy()
        X_test_scaled = X_test.copy()
        
        if n_features > 1:
            # normalize hours feature only
            hours_train = X_train[:, :, 1].flatten().reshape(-1, 1)
            hours_scaler = StandardScaler()
            hours_scaler.fit(hours_train)
            
            X_train_scaled[:, :, 1] = hours_scaler.transform(
                X_train[:, :, 1].reshape(-1, 1)
            ).reshape(n_train, context_len)
            X_val_scaled[:, :, 1] = hours_scaler.transform(
                X_val[:, :, 1].reshape(-1, 1)
            ).reshape(n_val, context_len)
            X_test_scaled[:, :, 1] = hours_scaler.transform(
                X_test[:, :, 1].reshape(-1, 1)
            ).reshape(n_test, context_len)
        
        # no scaling
        y_train_scaled = y_train
        y_val_scaled = y_val
        y_test_scaled = y_test
        scaler_y = None
    
    return (X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled,
            X_test_scaled, y_test_scaled, scaler_y)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 50,
    print_every: int = 5,
    patience: int = 10,
) -> nn.Module:
    model = model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        if (epoch + 1) % print_every == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], "
                  f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # early stop
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
    
    return model


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    scaler_y,
    device: torch.device,
) -> dict:
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            
            preds = outputs.cpu().numpy()
            targets = batch_y.numpy()
            
            if scaler_y is not None:
                preds = scaler_y.inverse_transform(preds)
                targets = scaler_y.inverse_transform(targets)
            
            all_preds.append(preds)
            all_targets.append(targets)
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    preds_flat = all_preds.flatten()
    targets_flat = all_targets.flatten()
    
    mae = mean_absolute_error(targets_flat, preds_flat)
    mse = mean_squared_error(targets_flat, preds_flat)
    rmse = np.sqrt(mse)
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'predictions': all_preds,
        'targets': all_targets,
    }


def compute_naive_baseline_mae(X_test: np.ndarray, y_test: np.ndarray) -> float:
    last_values = X_test[:, -1, 0]
    horizon_len = y_test.shape[1]
    naive_preds = np.repeat(last_values.reshape(-1, 1), horizon_len, axis=1)
    naive_mae = np.mean(np.abs(y_test - naive_preds))
    return naive_mae


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("LSTM BASELINE")
    print(f"\nContext length: {args.context_length}")
    print(f"Horizon length: {args.horizon_length}")
    print(f"Stride: {args.stride}")
    print(f"Seed: {args.seed}")
    
    df = load_data(args.data)
    
    X, y, market_ids = create_sequences(
        df,
        context_length=args.context_length,
        horizon_length=args.horizon_length,
        stride=args.stride,
        use_hours=args.use_hours,
    )
    
    X_train, y_train, X_val, y_val, X_test, y_test = split_by_market(
        X, y, market_ids,
        train_split=args.train_split,
        val_split=args.val_split,
        seed=args.seed,
        processor_path=args.processor,
    )
    
    (X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled,
     X_test_scaled, y_test_scaled, scaler_y) = scale_data(
        X_train, y_train, X_val, y_val, X_test, y_test,
        normalize_prices=args.normalize_prices,
    )
    
    train_dataset = PriceDataset(X_train_scaled, y_train_scaled)
    val_dataset = PriceDataset(X_val_scaled, y_val_scaled)
    test_dataset = PriceDataset(X_test_scaled, y_test_scaled)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    input_size = X_train.shape[2]
    model = LSTMForecaster(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        forecast_horizon=args.horizon_length,
        dropout=args.dropout,
    )
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    print("\nTraining...")
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        print_every=5,
        patience=args.patience,
    )

    print("\nEvaluating on test set...")
    metrics = evaluate_model(model, test_loader, scaler_y, device)
    
    naive_mae = compute_naive_baseline_mae(X_test, y_test)
    scaled_mae = metrics['MAE'] / naive_mae
    
    print("RESULTS")
    print(f"\nMAE: {metrics['MAE']:.6f}")
    print(f"MSE: {metrics['MSE']:.6f}")
    print(f"RMSE: {metrics['RMSE']:.6f}")
    print("\n" + "-" * 60)
    print("SCALED MAE:")
    print("-" * 60)
    print(f"Naive Baseline MAE: {naive_mae:.6f}")
    print(f"LSTM MAE: {metrics['MAE']:.6f}")
    print(f"Scaled MAE: {scaled_mae:.4f}")
    if scaled_mae < 1.0:
        print(f" so LSTM beats naive baseline by {(1-scaled_mae)*100:.1f}%")
    else:
        print(f" so Naive baseline beats LSTM by {(scaled_mae-1)*100:.1f}%")
    
    if args.save_model:
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {
                'context_length': args.context_length,
                'horizon_length': args.horizon_length,
                'hidden_size': args.hidden_size,
                'num_layers': args.num_layers,
                'input_size': input_size,
            },
            'metrics': metrics,
            'scaled_mae': scaled_mae,
        }, args.save_model)
        print(f"\nModel saved to: {args.save_model}")
    
    return model, metrics, scaled_mae


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--data", type=str, required=True)
    
    parser.add_argument("--context-length", type=int, default=CONTEXT_LENGTH)
    parser.add_argument("--horizon-length", type=int, default=HORIZON_LENGTH)
    parser.add_argument("--stride", type=int, default=STRIDE)
    
    parser.add_argument("--train-split", type=float, default=TRAIN_SPLIT)
    parser.add_argument("--val-split", type=float, default=VAL_SPLIT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--processor", type=str, default=None)
    
    parser.add_argument("--use-hours", action="store_true", default=True)
    parser.add_argument("--no-hours", dest="use_hours", action="store_false")
    parser.add_argument("--normalize-prices", action="store_true", default=False)
    
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    parser.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    parser.add_argument("--dropout", type=float, default=0.2)
    
    # training params
    parser.add_argument("--epochs", type=int, default=50,)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=10)
    
    parser.add_argument("--save-model", type=str, default=None)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
