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

## LCEVC

LCEVC has been added to the full VPCC pipeline, with changes in `vpcc_encoder/src/pipeline.rs`, `vpcc_decoder/src/main.rs`, and `gaussian_packing/src/lib.rs`.

Currently, the performance is similar to VPCC on it's own, with a slightly better compression ratio. Currently, LCEVC is set to always run, but that can be changed by modifying 

```
storage_config.lcevc = Some(LcevcConfig {
        enabled: true,
        downscale_factor: 2,
    });
```

and setting the `enabled` value to false.

I tried adding bilinear upscaling but saw no noticable size difference in the resulting .ply file. I sugggest exploring improved quantization, improving the downscaling, and combining the residual streams so there is only one large buffer. It might also be worth trying this implementation with mutlitple full codecs rather than the mock codec that VPCC and LCEVC are currently using in this pipeline.

Adding LCEVC might not be a the best approach in all situations, which has not yet been explored in this implementation. All testing has been done with point clouds.