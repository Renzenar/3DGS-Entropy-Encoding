# VPCC Prototype V1

This is the experimental raster-packed path. It does not replace the existing `encoder` / `decoder` and does not use a real video codec yet.

## Commands

Encode a PLY into a directory container:

```bash
cargo run -p vpcc_encoder -- <input.ply> <output_dir> [atlas_width] [compare_gsz]
```

Decode that directory container back into a PLY:

```bash
cargo run -p vpcc_decoder -- <output_dir> <reconstructed.ply>
```

## Output Layout

The encoder writes:

- `meta.json`
- `occupancy.bin`
- `geom_x.bin`
- `geom_y.bin`
- `geom_z.bin`
- `sh_dc_r.bin`
- `sh_dc_g.bin`
- `sh_dc_b.bin`

If `VPCC_WRITE_PREVIEWS=1` is set, it also writes debug-only normalized PNG previews for each plane.

## Diagnostics

The encoder prints:

- `XYZ RMSE`: quantization error after pack -> unpack for geometry
- `SH_DC RMSE`: quantization error after pack -> unpack for SH DC
- `Neighbor delta stats`: mean and standard deviation of adjacent differences in Morton order
- `Per-plane stats`:
  - `min` / `max`: occupied sample range in the packed plane
  - `entropy_bits`: simple Shannon entropy estimate on occupied samples
  - `delta_mean_abs` / `delta_std_abs`: adjacent sample variability in packed order
  - `h_corr` / `v_corr`: horizontal and vertical occupied-neighbor Pearson correlation
- `Total raw payload bytes`: bytes used by the canonical `.bin` planes
- `Original PLY size`: source file size on disk
- `Optional .gsz size`: only printed when a comparison path is provided

Higher neighbor correlation, lower delta magnitude, and lower entropy are the main early signals that raster packing may respond well to later video coding.
