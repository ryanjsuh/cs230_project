# Mirae

A foundation model for zero-shot prediction market forecasting. A CS 230 project.

## Getting Started

```bash
# for data processing, run the following:
cd polymarket-data

python -m venv venv
source venv/bin/activate

pip install -r requirements-dev.txt

python scripts/01_fetch_markets.py
python scripts/02_fetch_history.py
python scripts/03_clean_all.py
```

## Background

Prediction markets like Polymarket aggregate collective intelligence by enabling users to trade shares on real-world event outcomes, with prices reflecting probability estimates. While these markets synthesize information well at any moment, forecasting how probabilities evolve over time is challenging: traditional time-series methods struggle with bounded, non-stationary trajectories, and per-market models suffer from data scarcity. Inspired by foundation models like Google's TimesFM, this project trains a decoder-only Transformer on thousands of resolved Polymarket markets to learn universal patterns in crowd belief-updating, which enables zero-shot forecasting on newly listed markets using only past prices and basic metadata.

## Authors

[Aaron Jin](https://github.com/aaronkjin)

[Ryan Suh](https://github.com/ryanjsuh)
