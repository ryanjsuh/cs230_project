"""
Script to concatenate all checkpoint Parquet files into a single file
"""

import pandas as pd
from pathlib import Path


# Concatenate all checkpoint files into a single file
def concat_checkpoints(data_dir: str | Path = None, output_name: str = "data.parquet") -> Path:
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data"
    else:
        data_dir = Path(data_dir)
    
    # Find all checkpoint files
    checkpoint_files = sorted(data_dir.glob("checkpoint_*.parquet"))
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {data_dir}")
    
    print(f"Found {len(checkpoint_files)} checkpoint files:")
    for f in checkpoint_files:
        print(f"  - {f.name}")
    
    # Read and concatenate all files
    dfs = []
    for f in checkpoint_files:
        df = pd.read_parquet(f)
        print(f"  Loaded {f.name}: {len(df)} rows")
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal combined rows: {len(combined)}")
    
    # Write to output file
    output_path = data_dir / output_name
    combined.to_parquet(output_path, index=False)
    print(f"Saved to: {output_path}")
    
    return output_path


if __name__ == "__main__":
    concat_checkpoints()

