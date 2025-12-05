"""
Training script for model
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.config import ModelConfig, TrainingConfig, DataConfig
from model.model import PredictionMarketTimesFM, create_model
from model.dataset import DataProcessor, create_dataloaders


# Trainer class for model
class Trainer:
    def __init__(
        self,
        model: PredictionMarketTimesFM,
        train_loader,
        val_loader,
        config: TrainingConfig,
        processor: DataProcessor,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.processor = processor
        
        # Device
        self.device = torch.device(
            config.device if torch.cuda.is_available() and config.device == "cuda" 
            else "cpu"
        )
        print(f"Using device: {self.device}")
        self.model = self.model.to(self.device)
        
        # Loss function (MSE on price predictions)
        self.criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Learning rate scheduler
        if config.scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config.num_epochs,
                eta_min=config.learning_rate / 100,
            )
        elif config.scheduler == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True,
            )
        else:
            self.scheduler = None
        
        # Mixed precision
        self.scaler = GradScaler() if config.mixed_precision and self.device.type == "cuda" else None
        
        # Tracking
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.current_epoch = 0
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Train for one epoch
    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            context_patches = batch['context_patches'].to(self.device)
            target_patches = batch['target_patches'].to(self.device)
            category_ids = batch['category_id'].to(self.device)
            
            # Create mask
            batch_size, num_patches, patch_len, _ = context_patches.shape
            mask = torch.zeros(batch_size, num_patches, patch_len, 
                             dtype=torch.bool, device=self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with optional mixed precision
            if self.scaler is not None:
                with autocast():
                    # Model returns (predictions, stats, caches)
                    predictions, _, _ = self.model(context_patches, mask, category_ids)
                    # Loss on the last N patches (horizon prediction)
                    # predictions: (batch, num_patches, output_patch_len)
                    # target_patches: (batch, horizon_patches, output_patch_len)
                    horizon_patches = target_patches.size(1)
                    pred_horizon = predictions[:, -horizon_patches:, :]
                    loss = self.criterion(pred_horizon, target_patches)
                
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.grad_clip
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions, _, _ = self.model(context_patches, mask, category_ids)
                horizon_patches = target_patches.size(1)
                pred_horizon = predictions[:, -horizon_patches:, :]
                loss = self.criterion(pred_horizon, target_patches)
                
                loss.backward()
                
                if self.config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.grad_clip
                    )
                
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Logging
            if batch_idx % self.config.log_every == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.6f}")
        
        return total_loss / num_batches
    
    # Validate on validation set
    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        num_batches = 0
        
        for batch in self.val_loader:
            context_patches = batch['context_patches'].to(self.device)
            target_patches = batch['target_patches'].to(self.device)
            category_ids = batch['category_id'].to(self.device)
            
            # Create mask
            batch_size, num_patches, patch_len, _ = context_patches.shape
            mask = torch.zeros(batch_size, num_patches, patch_len, 
                             dtype=torch.bool, device=self.device)
            
            # Returns (predictions, stats, caches)
            predictions, _, _ = self.model(context_patches, mask, category_ids)
            horizon_patches = target_patches.size(1)
            pred_horizon = predictions[:, -horizon_patches:, :]
            
            loss = self.criterion(pred_horizon, target_patches)
            total_loss += loss.item()
            num_batches += 1
            
            # Collect for metrics
            all_preds.append(pred_horizon.cpu().numpy())
            all_targets.append(target_patches.cpu().numpy())
        
        # Compute metrics
        all_preds = np.concatenate(all_preds, axis=0).flatten()
        all_targets = np.concatenate(all_targets, axis=0).flatten()
        
        # Inverse transform to original scale for interpretable metrics
        all_preds_orig = self.processor.inverse_transform_prices(all_preds)
        all_targets_orig = self.processor.inverse_transform_prices(all_targets)
        
        mae = np.mean(np.abs(all_preds_orig - all_targets_orig))
        mse = np.mean((all_preds_orig - all_targets_orig) ** 2)
        rmse = np.sqrt(mse)
        
        return {
            'loss': total_loss / num_batches,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
        }
    
    # Save checkpoint
    def save_checkpoint(self, filename: str, is_best: bool = False):
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config,
        }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
        
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
    
    # Load checkpoint
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        print(f"Loaded checkpoint from {path}, epoch {self.current_epoch}")
    
    # Full training loop
    def train(self):
        print(f"\n{'='*60}")
        print("Starting training")
        print(f"{'='*60}")
        print(f"Model parameters: {self.model.count_parameters():,}")
        print(f"Training samples: {len(self.train_loader.dataset):,}")
        print(f"Validation samples: {len(self.val_loader.dataset):,}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"{'='*60}\n")
        
        patience = 10
        patience_counter = 0
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            print("-" * 40)
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_metrics = self.validate()
            val_loss = val_metrics['loss']
            self.val_losses.append(val_loss)
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Timing
            epoch_time = time.time() - epoch_start
            
            # Logging
            lr = self.optimizer.param_groups[0]['lr']
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Val Loss: {val_loss:.6f}")
            print(f"Val MAE: {val_metrics['mae']:.6f}")
            print(f"Val RMSE: {val_metrics['rmse']:.6f}")
            print(f"LR: {lr:.2e}")
            print(f"Time: {epoch_time:.1f}s")
            
            # Checkpointing
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                patience_counter = 0
                print("** New best model! **")
            else:
                patience_counter += 1
            
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt', is_best=is_best)
            elif is_best:
                self.save_checkpoint('best_model.pt', is_best=True)
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping after {patience} epochs without improvement")
                break
        
        # Final save
        self.save_checkpoint('final_model.pt')
        
        print(f"\n{'='*60}")
        print("Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Train TimesFM-inspired model")
    parser.add_argument("--data", type=str, required=True, help="Path to parquet data")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Model config overrides
    parser.add_argument("--d-model", type=int, default=128, help="Model dimension")
    parser.add_argument("--n-layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--n-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--patch-len", type=int, default=8, help="Input patch length")
    parser.add_argument("--output-patch-len", type=int, default=16, help="Output patch length")
    parser.add_argument("--context-patches", type=int, default=4, help="Number of context patches")
    parser.add_argument("--horizon-patches", type=int, default=2, help="Number of horizon patches")
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create configs
    model_config = ModelConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        input_patch_len=args.patch_len,
        output_patch_len=args.output_patch_len,
        context_patches=args.context_patches,
        horizon_patches=args.horizon_patches,
    )
    
    data_config = DataConfig(
        context_length=model_config.context_length,
        horizon_length=model_config.horizon_length,
        stride=args.patch_len,  
    )
    
    training_config = TrainingConfig(
        data_path=args.data,
        checkpoint_dir=args.checkpoint_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        seed=args.seed,
    )
    
    print(f"Model config: context={model_config.context_length}, horizon={model_config.horizon_length}")
    print(f"Patches: {model_config.context_patches} context x {model_config.input_patch_len}, "
          f"{model_config.horizon_patches} horizon x {model_config.output_patch_len}")
    
    # Process data
    processor = DataProcessor(model_config, data_config)
    train_dataset, val_dataset, test_dataset = processor.load_and_process(
        args.data,
        train_split=training_config.train_split,
        val_split=training_config.val_split,
        seed=args.seed,
    )
    
    # Save processor for later
    processor.save(Path(args.checkpoint_dir) / 'processor.pkl')
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=args.batch_size,
        num_workers=training_config.num_workers,
    )
    
    # Create model
    model = create_model(model_config)
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, training_config, processor)
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()
    
    # Final evaluation on test set
    print("\n" + "="*60)
    print("Evaluating on test set...")
    trainer.val_loader = test_loader
    test_metrics = trainer.validate()
    print(f"Test Loss: {test_metrics['loss']:.6f}")
    print(f"Test MAE: {test_metrics['mae']:.6f}")
    print(f"Test MSE: {test_metrics['mse']:.6f}")
    print(f"Test RMSE: {test_metrics['rmse']:.6f}")
    print("="*60)


if __name__ == "__main__":
    main()

