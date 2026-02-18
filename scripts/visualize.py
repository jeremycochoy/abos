#!/usr/bin/env python3
"""Visualize L1 snapshots from a Parquet file with 5-minute resampling.

Usage:
    python scripts/visualize.py [l1_snapshots.parquet]

Reads the L1 snapshots Parquet file produced by the simulation runner,
resamples to 5-minute bars, and plots bid/ask prices, spread, and volumes.
"""
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def load_l1_snapshots(path: str) -> pd.DataFrame:
    """Load L1 snapshots from Parquet and normalize prices/volumes."""
    df = pd.read_parquet(path)

    # Read tick_size and lot_size from Parquet metadata
    import pyarrow.parquet as pq
    meta = pq.read_schema(path).metadata
    tick_size = int(meta[b"tick_size"]) if b"tick_size" in meta else 1
    lot_size = int(meta[b"lot_size"]) if b"lot_size" in meta else 1

    # Convert timestamp (nanoseconds) to datetime
    df["time"] = pd.to_datetime(df["timestamp"], unit="ns")
    df = df.set_index("time")

    # Normalize prices and volumes
    df["bid_price"] = df["bid_price"].astype(float) / tick_size
    df["ask_price"] = df["ask_price"].astype(float) / tick_size
    df["bid_volume"] = df["bid_volume"].astype(float) / lot_size
    df["ask_volume"] = df["ask_volume"].astype(float) / lot_size

    # Filter out rows where bid or ask is 0 (empty book)
    df = df[(df["bid_price"] > 0) & (df["ask_price"] > 0)]

    # Derived columns
    df["mid_price"] = (df["bid_price"] + df["ask_price"]) / 2
    df["spread"] = df["ask_price"] - df["bid_price"]

    return df


def resample_5min(df: pd.DataFrame) -> pd.DataFrame:
    """Resample to 5-minute bars."""
    resampled = df.resample("300s").agg({
        "bid_price": "last",
        "ask_price": "last",
        "bid_volume": "sum",
        "ask_volume": "sum",
        "mid_price": "last",
        "spread": "mean",
    }).dropna()
    return resampled


def plot_l1(df: pd.DataFrame, output: str | None = None):
    """Create a 4-panel figure showing prices, spread, volume, and returns."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("L1 Market Data (5-min resampling)", fontsize=14)

    # Panel 1: Mid price
    ax = axes[0]
    ax.plot(df.index, df["mid_price"], linewidth=0.8, color="steelblue")
    ax.fill_between(df.index, df["bid_price"], df["ask_price"],
                     alpha=0.2, color="steelblue", label="Bid-Ask")
    ax.set_ylabel("Price ($)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Spread
    ax = axes[1]
    ax.plot(df.index, df["spread"], linewidth=0.8, color="coral")
    ax.set_ylabel("Spread ($)")
    ax.grid(True, alpha=0.3)

    # Panel 3: Volume
    ax = axes[2]
    ax.bar(df.index, df["bid_volume"], width=0.003, alpha=0.6,
           color="green", label="Bid volume")
    ax.bar(df.index, df["ask_volume"], width=0.003, alpha=0.6,
           color="red", label="Ask volume")
    ax.set_ylabel("Volume (lots)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: Log returns
    ax = axes[3]
    log_returns = df["mid_price"].apply(lambda x: x if x > 0 else float("nan")).apply(
        lambda x: float("nan") if pd.isna(x) else x
    )
    log_returns = log_returns.dropna().apply(lambda x: x).pct_change().dropna()
    ax.plot(log_returns.index, log_returns.values, linewidth=0.5,
            color="purple", alpha=0.7)
    ax.set_ylabel("Returns")
    ax.set_xlabel("Time")
    ax.grid(True, alpha=0.3)

    # Format x-axis
    for a in axes:
        a.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.setp(a.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output}")
    else:
        plt.show()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "l1_snapshots.parquet"
    if not Path(path).exists():
        print(f"Error: {path} not found. Run a simulation first.")
        sys.exit(1)

    print(f"Loading {path}...")
    df = load_l1_snapshots(path)
    print(f"  Loaded {len(df)} snapshots")
    print(f"  Time range: {df.index.min()} to {df.index.max()}")
    print(f"  Price range: ${df['mid_price'].min():.2f} - ${df['mid_price'].max():.2f}")

    print("Resampling to 5-minute bars...")
    resampled = resample_5min(df)
    print(f"  {len(resampled)} bars")

    output = path.replace(".parquet", ".png") if "--save" in sys.argv else None
    plot_l1(resampled, output)


if __name__ == "__main__":
    main()
