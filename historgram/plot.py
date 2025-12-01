import os
import matplotlib.pyplot as plt

def histogram_u32_csv_one_row(csv_path, num_bins=(1 << 20), max_val=(1 << 23)):
    """
    Reads a CSV file where millions of integer values (0..max_val) are stored on ONE row,
    comma-separated, and builds a histogram with `num_bins` bins over [0, max_val].
    """
    # Precompute scale factor: maps [0, max_val] -> [0, num_bins-1]
    scale = num_bins / (max_val + 1)

    bins = [0] * num_bins

    with open(csv_path, "r") as f:
        line = f.readline().strip()

        for token in line.split(","):
            token = token.strip()
            if not token:
                continue
            val = int(token)

            # Optional: sanity check
            if not (0 <= val <= max_val):
                # You can choose to clamp, skip, or raise. Here we clamp:
                val = max(0, min(val, max_val))

            idx = int(val * scale)
            if idx >= num_bins:
                idx = num_bins - 1  # safety
            bins[idx] += 1

    return bins

def plot_histogram_bins(bins, max_val, log_y=True):
    """
    Plot the histogram given an array of bin counts.
    x-axis: bin index mapped back to value range
    y-axis: count in each bin
    """
    num_bins = len(bins)
    # Compute approximate value range per bin for labeling
    bin_width = (max_val + 1) / num_bins
    x = [i * bin_width for i in range(num_bins)]

    plt.figure(figsize=(12, 6))
    plt.bar(x, bins, width=bin_width)
    plt.xlabel(f"Value range (0 .. {max_val})")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of values (binned into {num_bins} bins)")
    if log_y:
        plt.yscale("log")  # often helpful with skewed distributions
    plt.tight_layout()
    plt.show()

def main():
    csv_path = input("Enter path to CSV file containing the values: ").strip()

    if not os.path.isfile(csv_path):
        print(f"Error: File does not exist: {csv_path}")
        return

    # You can tweak these:
    max_val = (1 << 23)      # your stated max value
    num_bins = (1 << 20)          # increase or decrease for more/less granularity

    print("Reading CSV and computing histogram...")
    bins = histogram_u32_csv_one_row(csv_path, num_bins=num_bins, max_val=max_val)

    print("Plotting...")
    plot_histogram_bins(bins, max_val=max_val, log_y=True)

if __name__ == "__main__":
    main()

