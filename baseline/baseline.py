#!/usr/bin/env python3
"""
Baseline LSTM Forecaster for Polymarket Price Prediction

This script trains an LSTM model to forecast prediction market prices.
It uses a lookback window of 20 timesteps to predict the next 10 timesteps.

Usage:
    python baseline.py --data /path/to/polymarket_data.parquet
    python baseline.py --data /path/to/polymarket_data.parquet --epochs 30 --batch-size 128
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

LOOKBACK = 20
FORECAST = 10
HIDDEN_SIZE = 64
NUM_LAYERS = 2
RANDOM_STATE = 25

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
        forecast_horizon: int = 10
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, forecast_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # Take the last timestep
        output = self.fc(last_hidden)
        return output

def load_data(data_path: str) -> pd.DataFrame:
    print(f"Loading data from: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['won'].value_counts()}")
    return df

def create_sequences(
    df: pd.DataFrame,
    lookback: int = LOOKBACK,
    forecast: int = FORECAST
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # sort by market, token, and time
    df = df.sort_values(['market_id', 'token_id', 'timestamp'])
    
    sequences = []
    targets = []
    market_ids = []
    
    for (market_id, token_id), group in df.groupby(['market_id', 'token_id']):
        prices = group['price'].values
        hours_to_res = group['hours_to_resolution'].values
        
        # sliding windows
        for i in range(len(prices) - lookback - forecast + 1):
            seq_prices = prices[i:i + lookback]
            seq_hours = hours_to_res[i:i + lookback]
            
            # stack features
            sequence = np.column_stack([seq_prices, seq_hours])
            target = prices[i + lookback:i + lookback + forecast]
            
            sequences.append(sequence)
            targets.append(target)
            market_ids.append(market_id)
    
    X = np.array(sequences)
    y = np.array(targets)
    market_ids = np.array(market_ids)
    
    print(f"Total sequences: {len(X)}")
    print(f"Sequence shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    return X, y, market_ids

def split_by_market(
    X: np.ndarray,
    y: np.ndarray,
    market_ids: np.ndarray,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    unique_markets = np.unique(market_ids)
    train_markets, test_markets = train_test_split(
        unique_markets,
        test_size=test_size,
        random_state=random_state
    )
    
    train_mask = np.isin(market_ids, train_markets)
    test_mask = np.isin(market_ids, test_markets)
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"Training sequences: {len(X_train)}")
    print(f"Test sequences: {len(X_test)}")
    print(f"Training markets: {len(train_markets)}")
    print(f"Test markets: {len(test_markets)}")
    
    return X_train, y_train, X_test, y_test


def scale_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
    n_train, lookback, n_features = X_train.shape
    n_test = len(X_test)
    
    X_train_flat = X_train.reshape(-1, n_features)
    X_test_flat = X_test.reshape(-1, n_features)
    
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train_flat).reshape(n_train, lookback, n_features)
    X_test_scaled = scaler_X.transform(X_test_flat).reshape(n_test, lookback, n_features)
    
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, scaler_X, scaler_y

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 20,
    print_every: int = 5
) -> nn.Module:
    model = model.to(device)
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        if (epoch + 1) % print_every == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.6f}")
    
    return model

def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    scaler_y: StandardScaler,
    device: torch.device
) -> dict:
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            
            preds = scaler_y.inverse_transform(outputs.cpu().numpy())
            targets = scaler_y.inverse_transform(batch_y.numpy())
            
            all_preds.append(preds)
            all_targets.append(targets)
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    preds_flat = all_preds.flatten()
    targets_flat = all_targets.flatten()
    
    mae = mean_absolute_error(targets_flat, preds_flat)
    mse = mean_squared_error(targets_flat, preds_flat)
    rmse = np.sqrt(mse)
    
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse}

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    df = load_data(args.data)
    
    X, y, market_ids = create_sequences(df, lookback=LOOKBACK, forecast=FORECAST)
    
    X_train, y_train, X_test, y_test = split_by_market(X, y, market_ids)
    
    X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, scaler_X, scaler_y = scale_data(
        X_train, y_train, X_test, y_test
    )
    
    train_dataset = PriceDataset(X_train_scaled, y_train_scaled)
    test_dataset = PriceDataset(X_test_scaled, y_test_scaled)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = LSTMForecaster(
        input_size=2,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        forecast_horizon=FORECAST
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    model = train_model(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        print_every=5
    )

    metrics = evaluate_model(model, test_loader, scaler_y, device)
    
    print(f"MAE: {metrics['MAE']:.6f}")
    print(f"MSE: {metrics['MSE']:.6f}")
    print(f"RMSE: {metrics['RMSE']:.6f}")
    
    if args.save_model:
        torch.save(model.state_dict(), args.save_model)
        print(f"\nModel saved to: {args.save_model}")
    
    return model, metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train LSTM baseline model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the polymarket_data.parquet file"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--save-model",
        type=str,
        default=None,
        help="Path to save the trained model (optional)"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)

