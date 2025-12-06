# Mirae

A foundation model for zero-shot prediction market forecasting. A CS 230 project.

## Getting Started

For data processing, run the following:

```bash
cd polymarket-data

python -m venv venv
source venv/bin/activate

pip install -r requirements-dev.txt

# set bucket
export POLYMARKET_S3_BUCKET=cs230-polymarket-data-1

# run scripts in this order
python scripts/01_fetch_markets.py --s3 --all-markets
python scripts/02_fetch_history.py --s3
python scripts/03_clean_all.py --s3
```

To run the baseline, open and run all cells in the Google Colab:

```bash
# go into baseline-model directory, and run the cells in:
baseline.ipynb
```

To train the TimesFM-inspired decoder-only Transformer:

```bash
cd model

# install deps
pip install -r requirements.txt

# train (adjust paths below)
python train.py \
    --data ../polymarket-data/data/processed/data.parquet \
    --checkpoint-dir checkpoints \
    --epochs 50 \
    --batch-size 64 \
    --lr 1e-4 \
    --d-model 128 \
    --n-layers 4 \
    --n-heads 4

# evaluate on held-out markets
python evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --processor checkpoints/processor.pkl \
    --data ../polymarket-data/data/processed/data.parquet
```

## Background

Prediction markets like Polymarket aggregate collective intelligence by enabling users to trade shares on real-world event outcomes, with prices reflecting probability estimates. While these markets synthesize information well at any moment, forecasting how probabilities evolve over time is challenging: traditional time-series methods struggle with bounded, non-stationary trajectories, and per-market models suffer from data scarcity. Inspired by foundation models like Google's TimesFM, this project trains a decoder-only Transformer on thousands of resolved Polymarket markets to learn universal patterns in crowd belief-updating, which enables zero-shot forecasting on newly listed markets using only past prices and basic metadata.

### Model Architecture

Our model is a decoder-only Transformer inspired by [Google's TimesFM](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/). It uses patch-based tokenization, dividing price sequences into patches that group consecutive time points together. The architecture employs causal self-attention in a decoder-only setup, applying masked attention over these patches. Auxiliary features such as hours-to-resolution and category embeddings provide additional domain-specific context. The model's outputs are passed through a sigmoid activation to ensure predictions are bounded between [0, 1], and it generates future patches in an autoregressive manner, predicting one patch at a time.

Compared to the original TimesFM architecture, our model introduces several key modifications tailored for the prediction market setting. First, it applies a domain-specific sigmoid activation to ensure that outputs represent valid probabilities, naturally bounded between 0 and 1. Second, it encodes time-to-resolution information, allowing the model to capture crucial temporal dynamics as each market approaches settlement. Finally, the model leverages learned category embeddings to condition on market type, enabling it to generalize more effectively across diverse event domains.

## Authors

[Aaron Jin](https://github.com/aaronkjin)

[Ryan Suh](https://github.com/ryanjsuh)
