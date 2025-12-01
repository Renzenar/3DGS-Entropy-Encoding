#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt

MAX_BITS = 23
MAX_SYMBOL = (1 << MAX_BITS) - 1  # 0 .. 2^23 - 1


def load_values(csv_path: str) -> np.ndarray:
    """
    Load a single-line CSV of integers into a NumPy array.
    """
    values = np.fromfile(csv_path, sep=',', dtype=np.int64)
    if values.size == 0:
        raise ValueError("No values were read from the CSV.")
    return values


def build_histogram(values: np.ndarray, max_symbol: int = MAX_SYMBOL) -> np.ndarray:
    """
    Build a frequency array for symbols in [0, max_symbol].
    """
    if values.min() < 0 or values.max() > max_symbol:
        raise ValueError(
            f"Values out of expected range [0, {max_symbol}]. "
            f"Found min={values.min()}, max={values.max()}."
        )

    freq = np.bincount(values.astype(np.int64), minlength=max_symbol + 1)
    return freq


def plot_histogram(freq: np.ndarray, output_path: str) -> None:
    """
    Plot symbol value (x) vs frequency (y) and save to output_path.
    """
    x = np.nonzero(freq)[0]
    y = freq[x]

    if x.size == 0:
        raise ValueError("All frequencies are zero; nothing to plot.")

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, linewidth=0.5)
    plt.xlabel("Symbol value (integer in [0, 2^23 - 1])")
    plt.ylabel("Frequency")
    plt.title("Symbol Frequency Histogram")
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot histogram of symbol frequencies from a single-line CSV."
    )
    parser.add_argument("csv_path", help="Path to the CSV file.")
    parser.add_argument(
        "-o", "--output",
        default="histogram.png",
        help="Output image filename (default: histogram.png)",
    )
    args = parser.parse_args()

    print(f"Loading values from: {args.csv_path}")
    values = load_values(args.csv_path)

    count = values.size
    vmin = int(values.min())
    vmax = int(values.max())

    print(f"Loaded {count} values.")
    print(f"Minimum value: {vmin}")
    print(f"Maximum value: {vmax}")

    print(f"Building histogram over domain [0, {MAX_SYMBOL}]...")
    freq = build_histogram(values, MAX_SYMBOL)

    print("Plotting and saving...")
    plot_histogram(freq, args.output)


if __name__ == "__main__":
    main()

