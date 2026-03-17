use std::path::{Path, PathBuf};
use std::process::Command;

use gaussian_parser::load_gaussians_from_ply;
use gaussian_packing::{
    compute_roundtrip_diagnostics, evaluate_geom_hi_rgb_png_roundtrip, read_packed_raster_scene_dir,
    rebuild_packed_scene_with_decoded_geom_hi, load_geom_hi_rgb_png, sort_gaussians_by_morton,
    split_u16_plane_to_bytes, unpack_raster_to_scene, write_geom_hi_rgb_png, write_rgb_png_from_planes,
};
use gaussian_types::Gaussian;

const TILE_SIZE: usize = 8;

struct CodecRunResult {
    coded_path: PathBuf,
    decoded_png: PathBuf,
    coded_payload_bytes: u64,
    xyz_rmse: [f64; 3],
    additional_error: [f64; 3],
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input_ply = std::env::args().nth(1).expect("Missing input .ply path");
    let container_dir = std::env::args()
        .nth(2)
        .expect("Missing packed raster container directory");

    let scene = load_gaussians_from_ply(&input_ply)?;
    let sorted = sort_gaussians_by_morton(&scene);
    let packed = read_packed_raster_scene_dir(&container_dir)?;
    let baseline_scene = unpack_raster_to_scene(&packed);
    let baseline = compute_roundtrip_diagnostics(&sorted, &baseline_scene.gaussians, &packed);

    let work_dir = Path::new(&container_dir).join("geom_hi_codec_eval");
    std::fs::create_dir_all(&work_dir)?;

    let source_png = write_geom_hi_rgb_png(&packed, &work_dir)?.geom_hi_rgb;
    let lossless_rgb_png_bytes = std::fs::metadata(&source_png)?.len();
    let raw_geom_hi_bytes = packed.geom_x.len() as u64 * 3;

    let ffv1 = run_codec_roundtrip(
        &sorted,
        &packed,
        &baseline.xyz_rmse,
        &work_dir,
        &source_png,
        "ffv1_lossless",
        &[
            "-c:v",
            "ffv1",
            "-pix_fmt",
            "rgb24",
        ],
    )?;

    let x264rgb_crf18 = run_codec_roundtrip(
        &sorted,
        &packed,
        &baseline.xyz_rmse,
        &work_dir,
        &source_png,
        "libx264rgb_crf18",
        &[
            "-c:v",
            "libx264rgb",
            "-crf",
            "18",
            "-preset",
            "medium",
            "-pix_fmt",
            "rgb24",
        ],
    )?;

    let x264rgb_crf28 = run_codec_roundtrip(
        &sorted,
        &packed,
        &baseline.xyz_rmse,
        &work_dir,
        &source_png,
        "libx264rgb_crf28",
        &[
            "-c:v",
            "libx264rgb",
            "-crf",
            "28",
            "-preset",
            "medium",
            "-pix_fmt",
            "rgb24",
        ],
    )?;

    let predictive_source_png = write_predictive_geom_hi_horizontal_delta_png(&packed, &work_dir)?;
    let predictive_x264rgb_crf18 = run_predictive_codec_roundtrip(
        &sorted,
        &packed,
        &baseline.xyz_rmse,
        &work_dir,
        &predictive_source_png,
        "predictive_horizontal_delta_libx264rgb_crf18",
        &[
            "-c:v",
            "libx264rgb",
            "-crf",
            "18",
            "-preset",
            "medium",
            "-pix_fmt",
            "rgb24",
        ],
    )?;

    let tile_anchor_residual_x264rgb_crf18 = run_tile_anchor_residual_codec_roundtrip(
        &sorted,
        &packed,
        &baseline.xyz_rmse,
        &work_dir,
        "tile_anchor_residual_libx264rgb_crf18",
        TILE_SIZE,
        TILE_SIZE,
        &[
            "-c:v",
            "libx264rgb",
            "-crf",
            "18",
            "-preset",
            "medium",
            "-pix_fmt",
            "rgb24",
        ],
    )?;

    println!("Input PLY: {}", input_ply);
    println!("Container directory: {}", container_dir);
    println!("Work directory: {}", work_dir.display());
    println!("Source geom_hi RGB PNG: {}", source_png.display());
    println!("Raw geom hi payload bytes: {}", raw_geom_hi_bytes);
    println!("Lossless RGB PNG bytes: {}", lossless_rgb_png_bytes);
    println!(
        "Geometry RMSE quantization-only baseline: x={:.9}, y={:.9}, z={:.9}",
        baseline.xyz_rmse[0], baseline.xyz_rmse[1], baseline.xyz_rmse[2]
    );

    print_codec_result("Lossless FFV1", &ffv1, raw_geom_hi_bytes, lossless_rgb_png_bytes);
    print_codec_result("Lossy libx264rgb CRF18", &x264rgb_crf18, raw_geom_hi_bytes, lossless_rgb_png_bytes);
    print_codec_result("Lossy libx264rgb CRF28", &x264rgb_crf28, raw_geom_hi_bytes, lossless_rgb_png_bytes);
    print_codec_result(
        "Predictive horizontal-delta libx264rgb CRF18",
        &predictive_x264rgb_crf18,
        raw_geom_hi_bytes,
        lossless_rgb_png_bytes,
    );
    print_codec_result(
        "Tile-anchor residual libx264rgb CRF18",
        &tile_anchor_residual_x264rgb_crf18,
        raw_geom_hi_bytes,
        lossless_rgb_png_bytes,
    );

    println!("Conclusion:");
    println!(
        "Lossless RGB PNG remains the direct image baseline at {} bytes.",
        lossless_rgb_png_bytes
    );
    println!(
        "Lossless FFV1 payload is {} bytes.",
        ffv1.coded_payload_bytes
    );
    println!(
        "Lossy libx264rgb CRF18 payload is {} bytes with max additional RMSE {:.9}.",
        x264rgb_crf18.coded_payload_bytes,
        max_abs(&x264rgb_crf18.additional_error)
    );
    println!(
        "Lossy libx264rgb CRF28 payload is {} bytes with max additional RMSE {:.9}.",
        x264rgb_crf28.coded_payload_bytes,
        max_abs(&x264rgb_crf28.additional_error)
    );
    println!(
        "Predictive horizontal-delta libx264rgb CRF18 payload is {} bytes with max additional RMSE {:.9}.",
        predictive_x264rgb_crf18.coded_payload_bytes,
        max_abs(&predictive_x264rgb_crf18.additional_error)
    );
    println!(
        "Tile-anchor residual libx264rgb CRF18 payload is {} bytes with max additional RMSE {:.9}.",
        tile_anchor_residual_x264rgb_crf18.coded_payload_bytes,
        max_abs(&tile_anchor_residual_x264rgb_crf18.additional_error)
    );
    println!(
        "Predictive vs raw CRF18: byte_delta={} additional_rmse_delta={:.9}",
        predictive_x264rgb_crf18.coded_payload_bytes as i64 - x264rgb_crf18.coded_payload_bytes as i64,
        max_abs(&predictive_x264rgb_crf18.additional_error) - max_abs(&x264rgb_crf18.additional_error)
    );
    println!(
        "Tile-anchor residual vs raw CRF18: byte_delta={} additional_rmse_delta={:.9}",
        tile_anchor_residual_x264rgb_crf18.coded_payload_bytes as i64 - x264rgb_crf18.coded_payload_bytes as i64,
        max_abs(&tile_anchor_residual_x264rgb_crf18.additional_error)
            - max_abs(&x264rgb_crf18.additional_error)
    );

    Ok(())
}

fn run_codec_roundtrip(
    sorted: &[Gaussian],
    packed: &gaussian_packing::PackedRasterScene,
    baseline_xyz_rmse: &[f64; 3],
    work_dir: &Path,
    source_png: &Path,
    label: &'static str,
    codec_args: &[&str],
) -> Result<CodecRunResult, Box<dyn std::error::Error>> {
    let coded_path = work_dir.join(format!("{label}.mkv"));
    let decoded_dir = work_dir.join(format!("{label}_decoded"));
    std::fs::create_dir_all(&decoded_dir)?;
    let decoded_png = decoded_dir.join("geom_hi_rgb.png");

    let mut encode_args = vec![
        "-y".to_string(),
        "-i".to_string(),
        source_png.display().to_string(),
    ];
    encode_args.extend(codec_args.iter().map(|arg| (*arg).to_string()));
    encode_args.push(coded_path.display().to_string());
    run_ffmpeg(&encode_args)?;

    let decode_args = vec![
        "-y".to_string(),
        "-i".to_string(),
        coded_path.display().to_string(),
        "-frames:v".to_string(),
        "1".to_string(),
        decoded_png.display().to_string(),
    ];
    run_ffmpeg(&decode_args)?;

    let result = evaluate_geom_hi_rgb_png_roundtrip(sorted, packed, &decoded_dir)?;
    let coded_payload_bytes = std::fs::metadata(&coded_path)?.len();
    let additional_error = [
        result.xyz_rmse[0] - baseline_xyz_rmse[0],
        result.xyz_rmse[1] - baseline_xyz_rmse[1],
        result.xyz_rmse[2] - baseline_xyz_rmse[2],
    ];

    Ok(CodecRunResult {
        coded_path,
        decoded_png,
        coded_payload_bytes,
        xyz_rmse: result.xyz_rmse,
        additional_error,
    })
}

fn run_tile_anchor_residual_codec_roundtrip(
    sorted: &[Gaussian],
    packed: &gaussian_packing::PackedRasterScene,
    baseline_xyz_rmse: &[f64; 3],
    work_dir: &Path,
    label: &'static str,
    tile_width: usize,
    tile_height: usize,
    codec_args: &[&str],
) -> Result<CodecRunResult, Box<dyn std::error::Error>> {
    let artifacts = write_tile_anchor_residual_geom_hi_rgb(packed, work_dir, tile_width, tile_height)?;
    let coded_path = work_dir.join(format!("{label}.mkv"));
    let decoded_dir = work_dir.join(format!("{label}_decoded"));
    std::fs::create_dir_all(&decoded_dir)?;
    let decoded_png = decoded_dir.join("geom_hi_rgb.png");

    let mut encode_args = vec![
        "-y".to_string(),
        "-i".to_string(),
        artifacts.residual_png.display().to_string(),
    ];
    encode_args.extend(codec_args.iter().map(|arg| (*arg).to_string()));
    encode_args.push(coded_path.display().to_string());
    run_ffmpeg(&encode_args)?;

    let decode_args = vec![
        "-y".to_string(),
        "-i".to_string(),
        coded_path.display().to_string(),
        "-frames:v".to_string(),
        "1".to_string(),
        decoded_png.display().to_string(),
    ];
    run_ffmpeg(&decode_args)?;

    let decoded_residual = load_geom_hi_rgb_png(&decoded_dir, packed.width, packed.height)?;
    let decoded_hi = [
        tile_anchor_residual_decode_plane(
            &decoded_residual[0],
            &artifacts.geom_x_anchor,
            packed.width as usize,
            packed.height as usize,
            tile_width,
            tile_height,
        ),
        tile_anchor_residual_decode_plane(
            &decoded_residual[1],
            &artifacts.geom_y_anchor,
            packed.width as usize,
            packed.height as usize,
            tile_width,
            tile_height,
        ),
        tile_anchor_residual_decode_plane(
            &decoded_residual[2],
            &artifacts.geom_z_anchor,
            packed.width as usize,
            packed.height as usize,
            tile_width,
            tile_height,
        ),
    ];

    let rebuilt = rebuild_packed_scene_with_decoded_geom_hi(
        packed,
        [&decoded_hi[0], &decoded_hi[1], &decoded_hi[2]],
    );
    let reconstructed = unpack_raster_to_scene(&rebuilt);
    let xyz_rmse = [
        rmse_for_axis(sorted, &reconstructed.gaussians, 0),
        rmse_for_axis(sorted, &reconstructed.gaussians, 1),
        rmse_for_axis(sorted, &reconstructed.gaussians, 2),
    ];
    let additional_error = [
        xyz_rmse[0] - baseline_xyz_rmse[0],
        xyz_rmse[1] - baseline_xyz_rmse[1],
        xyz_rmse[2] - baseline_xyz_rmse[2],
    ];
    let coded_payload_bytes =
        std::fs::metadata(&coded_path)?.len() + artifacts.anchor_bytes() as u64;

    Ok(CodecRunResult {
        coded_path,
        decoded_png,
        coded_payload_bytes,
        xyz_rmse,
        additional_error,
    })
}

fn run_predictive_codec_roundtrip(
    sorted: &[Gaussian],
    packed: &gaussian_packing::PackedRasterScene,
    baseline_xyz_rmse: &[f64; 3],
    work_dir: &Path,
    source_png: &Path,
    label: &'static str,
    codec_args: &[&str],
) -> Result<CodecRunResult, Box<dyn std::error::Error>> {
    let coded_path = work_dir.join(format!("{label}.mkv"));
    let decoded_dir = work_dir.join(format!("{label}_decoded"));
    std::fs::create_dir_all(&decoded_dir)?;
    let decoded_png = decoded_dir.join("geom_hi_rgb.png");

    let mut encode_args = vec![
        "-y".to_string(),
        "-i".to_string(),
        source_png.display().to_string(),
    ];
    encode_args.extend(codec_args.iter().map(|arg| (*arg).to_string()));
    encode_args.push(coded_path.display().to_string());
    run_ffmpeg(&encode_args)?;

    let decode_args = vec![
        "-y".to_string(),
        "-i".to_string(),
        coded_path.display().to_string(),
        "-frames:v".to_string(),
        "1".to_string(),
        decoded_png.display().to_string(),
    ];
    run_ffmpeg(&decode_args)?;

    let decoded_predictive = load_geom_hi_rgb_png(&decoded_dir, packed.width, packed.height)?;
    let decoded_hi = [
        horizontal_delta_decode_plane(&decoded_predictive[0], packed.width, packed.height),
        horizontal_delta_decode_plane(&decoded_predictive[1], packed.width, packed.height),
        horizontal_delta_decode_plane(&decoded_predictive[2], packed.width, packed.height),
    ];
    let rebuilt = rebuild_packed_scene_with_decoded_geom_hi(
        packed,
        [&decoded_hi[0], &decoded_hi[1], &decoded_hi[2]],
    );
    let reconstructed = unpack_raster_to_scene(&rebuilt);

    let xyz_rmse = [
        rmse_for_axis(sorted, &reconstructed.gaussians, 0),
        rmse_for_axis(sorted, &reconstructed.gaussians, 1),
        rmse_for_axis(sorted, &reconstructed.gaussians, 2),
    ];
    let additional_error = [
        xyz_rmse[0] - baseline_xyz_rmse[0],
        xyz_rmse[1] - baseline_xyz_rmse[1],
        xyz_rmse[2] - baseline_xyz_rmse[2],
    ];
    let coded_payload_bytes = std::fs::metadata(&coded_path)?.len();

    Ok(CodecRunResult {
        coded_path,
        decoded_png,
        coded_payload_bytes,
        xyz_rmse,
        additional_error,
    })
}

fn print_codec_result(
    title: &str,
    result: &CodecRunResult,
    raw_geom_hi_bytes: u64,
    lossless_rgb_png_bytes: u64,
) {
    println!("{title}:");
    println!("  coded payload: {}", result.coded_path.display());
    println!("  decoded png: {}", result.decoded_png.display());
    println!("  coded payload bytes: {}", result.coded_payload_bytes);
    println!(
        "  coded/raw ratio: {:.6}",
        result.coded_payload_bytes as f64 / raw_geom_hi_bytes as f64
    );
    println!(
        "  coded/lossless-rgb-png ratio: {:.6}",
        result.coded_payload_bytes as f64 / lossless_rgb_png_bytes as f64
    );
    println!(
        "  geometry RMSE: x={:.9}, y={:.9}, z={:.9}",
        result.xyz_rmse[0], result.xyz_rmse[1], result.xyz_rmse[2]
    );
    println!(
        "  additional RMSE beyond baseline: x={:.9}, y={:.9}, z={:.9}",
        result.additional_error[0], result.additional_error[1], result.additional_error[2]
    );
}

fn max_abs(values: &[f64; 3]) -> f64 {
    values.iter().map(|value| value.abs()).fold(0.0_f64, f64::max)
}

struct TileAnchorResidualArtifacts {
    residual_png: PathBuf,
    geom_x_anchor: Vec<u8>,
    geom_y_anchor: Vec<u8>,
    geom_z_anchor: Vec<u8>,
}

impl TileAnchorResidualArtifacts {
    fn anchor_bytes(&self) -> usize {
        self.geom_x_anchor.len() + self.geom_y_anchor.len() + self.geom_z_anchor.len()
    }
}

fn write_predictive_geom_hi_horizontal_delta_png(
    packed: &gaussian_packing::PackedRasterScene,
    work_dir: &Path,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let predictive_dir = work_dir.join("predictive_horizontal_delta");
    std::fs::create_dir_all(&predictive_dir)?;
    let predictive_png = predictive_dir.join("geom_hi_rgb.png");

    let (geom_x_hi, _) = split_u16_plane_to_bytes(&packed.geom_x);
    let (geom_y_hi, _) = split_u16_plane_to_bytes(&packed.geom_y);
    let (geom_z_hi, _) = split_u16_plane_to_bytes(&packed.geom_z);

    let geom_x_pred = horizontal_delta_encode_plane(&geom_x_hi, packed.width, packed.height);
    let geom_y_pred = horizontal_delta_encode_plane(&geom_y_hi, packed.width, packed.height);
    let geom_z_pred = horizontal_delta_encode_plane(&geom_z_hi, packed.width, packed.height);

    write_rgb_png_from_planes(
        &predictive_png,
        packed.width,
        packed.height,
        &geom_x_pred,
        &geom_y_pred,
        &geom_z_pred,
    )?;

    Ok(predictive_png)
}

fn write_tile_anchor_residual_geom_hi_rgb(
    packed: &gaussian_packing::PackedRasterScene,
    work_dir: &Path,
    tile_width: usize,
    tile_height: usize,
) -> Result<TileAnchorResidualArtifacts, Box<dyn std::error::Error>> {
    let tile_dir = work_dir.join(format!("tile_anchor_residual_{}x{}", tile_width, tile_height));
    std::fs::create_dir_all(&tile_dir)?;
    let residual_png = tile_dir.join("geom_hi_rgb.png");

    let width = packed.width as usize;
    let height = packed.height as usize;
    let (geom_x_hi, _) = split_u16_plane_to_bytes(&packed.geom_x);
    let (geom_y_hi, _) = split_u16_plane_to_bytes(&packed.geom_y);
    let (geom_z_hi, _) = split_u16_plane_to_bytes(&packed.geom_z);

    let (geom_x_anchor, geom_x_residual) =
        tile_anchor_residual_encode_plane(&geom_x_hi, width, height, tile_width, tile_height);
    let (geom_y_anchor, geom_y_residual) =
        tile_anchor_residual_encode_plane(&geom_y_hi, width, height, tile_width, tile_height);
    let (geom_z_anchor, geom_z_residual) =
        tile_anchor_residual_encode_plane(&geom_z_hi, width, height, tile_width, tile_height);

    std::fs::write(tile_dir.join("geom_x_anchor.bin"), &geom_x_anchor)?;
    std::fs::write(tile_dir.join("geom_y_anchor.bin"), &geom_y_anchor)?;
    std::fs::write(tile_dir.join("geom_z_anchor.bin"), &geom_z_anchor)?;

    write_rgb_png_from_planes(
        &residual_png,
        packed.width,
        packed.height,
        &geom_x_residual,
        &geom_y_residual,
        &geom_z_residual,
    )?;

    Ok(TileAnchorResidualArtifacts {
        residual_png,
        geom_x_anchor,
        geom_y_anchor,
        geom_z_anchor,
    })
}

fn horizontal_delta_encode_plane(data: &[u8], width: u32, height: u32) -> Vec<u8> {
    let width = width as usize;
    let height = height as usize;
    let mut encoded = vec![0u8; data.len()];

    for y in 0..height {
        let row_start = y * width;
        if width == 0 || row_start >= data.len() {
            continue;
        }
        encoded[row_start] = data[row_start];
        for x in 1..width {
            let idx = row_start + x;
            if idx >= data.len() {
                break;
            }
            encoded[idx] = data[idx].wrapping_sub(data[idx - 1]);
        }
    }

    encoded
}

fn horizontal_delta_decode_plane(data: &[u8], width: u32, height: u32) -> Vec<u8> {
    let width = width as usize;
    let height = height as usize;
    let mut decoded = vec![0u8; data.len()];

    for y in 0..height {
        let row_start = y * width;
        if width == 0 || row_start >= data.len() {
            continue;
        }
        decoded[row_start] = data[row_start];
        for x in 1..width {
            let idx = row_start + x;
            if idx >= data.len() {
                break;
            }
            decoded[idx] = decoded[idx - 1].wrapping_add(data[idx]);
        }
    }

    decoded
}

fn tile_anchor_residual_encode_plane(
    data: &[u8],
    width: usize,
    height: usize,
    tile_width: usize,
    tile_height: usize,
) -> (Vec<u8>, Vec<u8>) {
    let tiles_x = width.div_ceil(tile_width);
    let tiles_y = height.div_ceil(tile_height);
    let mut anchors = Vec::with_capacity(tiles_x * tiles_y);
    let mut residuals = vec![0u8; data.len()];

    for tile_y in 0..tiles_y {
        for tile_x in 0..tiles_x {
            let x0 = tile_x * tile_width;
            let y0 = tile_y * tile_height;
            let x1 = (x0 + tile_width).min(width);
            let y1 = (y0 + tile_height).min(height);

            let mut min_val = u8::MAX;
            let mut max_val = u8::MIN;
            for y in y0..y1 {
                for x in x0..x1 {
                    let value = data[y * width + x];
                    min_val = min_val.min(value);
                    max_val = max_val.max(value);
                }
            }

            let anchor = ((min_val as u16 + max_val as u16 + 1) / 2) as u8;
            anchors.push(anchor);

            for y in y0..y1 {
                for x in x0..x1 {
                    let idx = y * width + x;
                    let residual = data[idx] as i16 - anchor as i16 + 128;
                    residuals[idx] = residual.clamp(0, 255) as u8;
                }
            }
        }
    }

    (anchors, residuals)
}

fn tile_anchor_residual_decode_plane(
    residuals: &[u8],
    anchors: &[u8],
    width: usize,
    height: usize,
    tile_width: usize,
    tile_height: usize,
) -> Vec<u8> {
    let tiles_x = width.div_ceil(tile_width);
    let tiles_y = height.div_ceil(tile_height);
    let mut decoded = vec![0u8; residuals.len()];
    let mut anchor_idx = 0usize;

    for tile_y in 0..tiles_y {
        for tile_x in 0..tiles_x {
            let anchor = anchors[anchor_idx] as i16;
            anchor_idx += 1;

            let x0 = tile_x * tile_width;
            let y0 = tile_y * tile_height;
            let x1 = (x0 + tile_width).min(width);
            let y1 = (y0 + tile_height).min(height);

            for y in y0..y1 {
                for x in x0..x1 {
                    let idx = y * width + x;
                    let value = anchor + residuals[idx] as i16 - 128;
                    decoded[idx] = value.clamp(0, 255) as u8;
                }
            }
        }
    }

    decoded
}

fn rmse_for_axis(original: &[Gaussian], reconstructed: &[Gaussian], axis: usize) -> f64 {
    if original.is_empty() {
        return 0.0;
    }

    let mse = original
        .iter()
        .zip(reconstructed.iter())
        .map(|(lhs, rhs)| {
            let diff = lhs.xyz[axis] as f64 - rhs.xyz[axis] as f64;
            diff * diff
        })
        .sum::<f64>()
        / original.len() as f64;

    mse.sqrt()
}

fn run_ffmpeg(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    println!("Running: ffmpeg {}", args.join(" "));
    let output = Command::new("ffmpeg").args(args).output().map_err(|err| {
        format!("failed to launch ffmpeg; ensure it is installed and on PATH: {err}")
    })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(format!(
            "ffmpeg command failed with status {}.\nstdout:\n{}\nstderr:\n{}",
            output.status, stdout, stderr
        )
        .into());
    }

    Ok(())
}
