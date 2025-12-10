# 3DGS Entropy Encoding

This project implements a custom entropy encoding and decoding pipeline optimized for 3D Gaussian Splatting (3DGS) data. It utilizes a block-adaptive rANS (Range Asymmetric Numeral Systems) coder to compress Gaussian attribute streams efficiently.

## Prerequisites

This project includes a `*-sys` crate that uses [`bindgen`](https://github.com/rust-lang/rust-bindgen) to generate Rust FFI bindings at build time.  
Because of this, you need:

- A working **Rust toolchain** (via [rustup](https://rustup.rs/))
- A **C/C++ build toolchain** (compiler, linker, `make`, etc.)
- **Clang/LLVM with libclang** available on your system

## Usage

### Encoder

The encoder reads a `.ply` file and outputs a compressed `.gsc` file.

```bash
cargo run --release -p encoder -- <input_path.ply> <output_path.gsc>
```

### Decoder

The decoder binary reads a compressed `.gsc` file and outputs a `.ply` file.

```bash
cargo run --release -p decoder -- <input_path.gsc> <output_path.ply>
```

## Pipeline

The system is designed around a specialized pipeline that transforms raw floating-point Gaussian attributes into a compressed binary format (`.gsc`) and reconstructs them back into PLY files.

### Core Components

1.  **Componentization & Quantization**:
    - Floating-point values (positions, scales, rotations, SH coefficients) are decomposed into three separate channels: **Sign**, **Exponent**, and **Mantissa**.
    - Mantissas are quantized based on a configurable step size to reduce entropy while maintaining visual fidelity.
    - High-error outliers in quantization are handled by falling back to raw storage to preserve precision where necessary.

2.  **Prediction & Delta Encoding**:
    - The pipeline applies delta encoding to the Exponent and Quantized Mantissa indices. This leverages the spatial coherence often found in sorted Gaussian splat data to minimize value variance before entropy coding.

3.  **Block-Adaptive rANS Coding**:
    - The project uses a multi-channel rANS encoder/decoder (`rans_coding`).
    - **Context Modeling**: The coder maintains adaptive frequency tables (histograms) for each channel (Sign, Exponent, Mantissa).
    - **Adaptivity**: The probability model is updated dynamically. When the total frequency count reaches a threshold (`scale_bit`), the histogram is rescaled, and a snapshot is taken (for the encoder) or used (by the decoder) to adapt to changing data distributions within the stream.
    - **Interleaving**: The encoder processes data in a LIFO (Last-In-First-Out) manner, while the decoder processes it forwards, necessitating a snapshotting mechanism in the encoder to match the decoder's state reconstruction.

### File Formats
-   **Input**: Standard Gaussian Splatting data (`.ply`) from [vanilla 3DGS](https://github.com/graphdeco-inria/gaussian-splatting): 
-   **Output (`.gsc`)**: A custom binary container consisting of:
    -   Total scene length (number of Gaussians).
    -   Blocks of rANS coded streams paired with raw escape data.
-   **Reconstruction**: Decoded data is exported to standard `.ply` format compatible with 3DGS viewers.

