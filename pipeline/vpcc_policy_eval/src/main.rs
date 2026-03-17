use std::path::{Path, PathBuf};
use std::process::Command;

use gaussian_parser::load_gaussians_from_ply;
use gaussian_packing::{
    compute_roundtrip_diagnostics, load_geom_hi_rgb_png, read_packed_raster_scene_dir,
    read_rgb_png_to_planes, rebuild_packed_scene_with_decoded_geom_hi,
    rebuild_packed_scene_with_decoded_sh_dc_hi, sort_gaussians_by_morton, split_u16_plane_to_bytes,
    unpack_raster_to_scene, write_geom_hi_rgb_png, write_rgb_png_from_planes, write_sh_dc_hi_rgb_png,
};
use gaussian_types::Gaussian;

const TILE_SIZE: usize = 8;

struct CodecRunResult {
    coded_path: PathBuf,
    decoded_png: PathBuf,
    coded_payload_bytes: u64,
    rmse: [f64; 3],
    additional_error: [f64; 3],
    additional_ratio: [f64; 3],
    psnr: [f64; 3],
    max_abs_error: [f64; 3],
    ssim: Option<f64>,
}

struct ShDcCodecArtifacts {
    coded_path: PathBuf,
    decoded_png: PathBuf,
    coded_payload_bytes: u64,
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mode = std::env::args().nth(1).expect("Missing mode: geom_hi | sh_dc_hi | mixed");
    let input_ply = std::env::args().nth(2).expect("Missing input .ply path");
    let container_dir = std::env::args()
        .nth(3)
        .expect("Missing packed raster container directory");

    match mode.as_str() {
        "geom_hi" => run_geom_hi_mode(&input_ply, &container_dir),
        "sh_dc_hi" => run_sh_dc_hi_mode(&input_ply, &container_dir),
        "mixed" => run_mixed_mode(&input_ply, &container_dir),
        _ => Err(format!("Unknown mode '{}'. Expected: geom_hi | sh_dc_hi | mixed", mode).into()),
    }
}

fn run_geom_hi_mode(input_ply: &str, container_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
    let scene = load_gaussians_from_ply(input_ply)?;
    let sorted = sort_gaussians_by_morton(&scene);
    let packed = read_packed_raster_scene_dir(container_dir)?;
    let baseline_scene = unpack_raster_to_scene(&packed);
    let baseline = compute_roundtrip_diagnostics(&sorted, &baseline_scene.gaussians, &packed);

    let work_dir = Path::new(container_dir).join("geom_hi_codec_eval");
    std::fs::create_dir_all(&work_dir)?;

    let source_png = write_geom_hi_rgb_png(&packed, &work_dir)?.geom_hi_rgb;
    let lossless_rgb_png_bytes = std::fs::metadata(&source_png)?.len();
    let raw_geom_hi_bytes = packed.geom_x.len() as u64 * 3;

    let ffv1 = run_geom_codec_roundtrip(
        &sorted,
        &packed,
        &baseline.xyz_rmse,
        &work_dir,
        &source_png,
        "ffv1_lossless",
        &["-c:v", "ffv1", "-pix_fmt", "rgb24"],
    )?;

    let x264rgb_crf18 = run_geom_codec_roundtrip(
        &sorted,
        &packed,
        &baseline.xyz_rmse,
        &work_dir,
        &source_png,
        "libx264rgb_crf18",
        &["-c:v", "libx264rgb", "-crf", "18", "-preset", "medium", "-pix_fmt", "rgb24"],
    )?;

    let x264rgb_crf28 = run_geom_codec_roundtrip(
        &sorted,
        &packed,
        &baseline.xyz_rmse,
        &work_dir,
        &source_png,
        "libx264rgb_crf28",
        &["-c:v", "libx264rgb", "-crf", "28", "-preset", "medium", "-pix_fmt", "rgb24"],
    )?;

    let predictive_source_png = write_predictive_geom_hi_horizontal_delta_png(&packed, &work_dir)?;
    let predictive_x264rgb_crf18 = run_predictive_geom_codec_roundtrip(
        &sorted,
        &packed,
        &baseline.xyz_rmse,
        &work_dir,
        &predictive_source_png,
        "predictive_horizontal_delta_libx264rgb_crf18",
        &["-c:v", "libx264rgb", "-crf", "18", "-preset", "medium", "-pix_fmt", "rgb24"],
    )?;

    let tile_anchor_residual_x264rgb_crf18 = run_tile_anchor_residual_codec_roundtrip(
        &sorted,
        &packed,
        &baseline.xyz_rmse,
        &work_dir,
        "tile_anchor_residual_libx264rgb_crf18",
        TILE_SIZE,
        TILE_SIZE,
        &["-c:v", "libx264rgb", "-crf", "18", "-preset", "medium", "-pix_fmt", "rgb24"],
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

    print_codec_result("Lossless FFV1", &ffv1, raw_geom_hi_bytes, lossless_rgb_png_bytes, ffv1.coded_payload_bytes, "geometry");
    print_codec_result("Lossy libx264rgb CRF18", &x264rgb_crf18, raw_geom_hi_bytes, lossless_rgb_png_bytes, ffv1.coded_payload_bytes, "geometry");
    print_codec_result("Lossy libx264rgb CRF28", &x264rgb_crf28, raw_geom_hi_bytes, lossless_rgb_png_bytes, ffv1.coded_payload_bytes, "geometry");
    print_codec_result(
        "Predictive horizontal-delta libx264rgb CRF18",
        &predictive_x264rgb_crf18,
        raw_geom_hi_bytes,
        lossless_rgb_png_bytes,
        ffv1.coded_payload_bytes,
        "geometry",
    );
    print_codec_result(
        "Tile-anchor residual libx264rgb CRF18",
        &tile_anchor_residual_x264rgb_crf18,
        raw_geom_hi_bytes,
        lossless_rgb_png_bytes,
        ffv1.coded_payload_bytes,
        "geometry",
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
    print_summary_block(
        "geom_hi",
        &[
            ("Lossless FFV1", &ffv1, raw_geom_hi_bytes),
            ("Lossy libx264rgb CRF18", &x264rgb_crf18, raw_geom_hi_bytes),
            ("Lossy libx264rgb CRF28", &x264rgb_crf28, raw_geom_hi_bytes),
            ("Predictive horizontal-delta libx264rgb CRF18", &predictive_x264rgb_crf18, raw_geom_hi_bytes),
            ("Tile-anchor residual libx264rgb CRF18", &tile_anchor_residual_x264rgb_crf18, raw_geom_hi_bytes),
        ],
        "geometry",
    );

    Ok(())
}

fn run_sh_dc_hi_mode(input_ply: &str, container_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
    let scene = load_gaussians_from_ply(input_ply)?;
    let sorted = sort_gaussians_by_morton(&scene);
    let packed = read_packed_raster_scene_dir(container_dir)?;
    let baseline_scene = unpack_raster_to_scene(&packed);
    let baseline = compute_roundtrip_diagnostics(&sorted, &baseline_scene.gaussians, &packed);

    let work_dir = Path::new(container_dir).join("sh_dc_hi_codec_eval");
    std::fs::create_dir_all(&work_dir)?;

    let source_png = write_sh_dc_hi_rgb_png(&packed, &work_dir)?.sh_dc_hi_rgb;
    let lossless_rgb_png_bytes = std::fs::metadata(&source_png)?.len();
    let raw_sh_dc_hi_bytes = packed.sh_dc_r.len() as u64 * 3;

    let ffv1 = run_sh_dc_codec_roundtrip(
        &sorted,
        &packed,
        &baseline.sh_dc_rmse,
        &work_dir,
        &source_png,
        "ffv1_lossless",
        &["-c:v", "ffv1", "-pix_fmt", "rgb24"],
    )?;

    let x264rgb_crf12 = run_sh_dc_codec_roundtrip(
        &sorted,
        &packed,
        &baseline.sh_dc_rmse,
        &work_dir,
        &source_png,
        "libx264rgb_crf12",
        &["-c:v", "libx264rgb", "-crf", "12", "-preset", "medium", "-pix_fmt", "rgb24"],
    )?;

    let x264rgb_crf18 = run_sh_dc_codec_roundtrip(
        &sorted,
        &packed,
        &baseline.sh_dc_rmse,
        &work_dir,
        &source_png,
        "libx264rgb_crf18",
        &["-c:v", "libx264rgb", "-crf", "18", "-preset", "medium", "-pix_fmt", "rgb24"],
    )?;

    let x264rgb_crf28 = run_sh_dc_codec_roundtrip(
        &sorted,
        &packed,
        &baseline.sh_dc_rmse,
        &work_dir,
        &source_png,
        "libx264rgb_crf28",
        &["-c:v", "libx264rgb", "-crf", "28", "-preset", "medium", "-pix_fmt", "rgb24"],
    )?;

    println!("Input PLY: {}", input_ply);
    println!("Container directory: {}", container_dir);
    println!("Work directory: {}", work_dir.display());
    println!("Source sh_dc_hi RGB PNG: {}", source_png.display());
    println!("Raw sh_dc hi payload bytes: {}", raw_sh_dc_hi_bytes);
    println!("Lossless RGB PNG bytes: {}", lossless_rgb_png_bytes);
    println!(
        "SH DC RMSE quantization-only baseline: r={:.9}, g={:.9}, b={:.9}",
        baseline.sh_dc_rmse[0], baseline.sh_dc_rmse[1], baseline.sh_dc_rmse[2]
    );

    print_codec_result("Lossless FFV1", &ffv1, raw_sh_dc_hi_bytes, lossless_rgb_png_bytes, ffv1.coded_payload_bytes, "sh_dc");
    print_codec_result("Lossy libx264rgb CRF12", &x264rgb_crf12, raw_sh_dc_hi_bytes, lossless_rgb_png_bytes, ffv1.coded_payload_bytes, "sh_dc");
    print_codec_result("Lossy libx264rgb CRF18", &x264rgb_crf18, raw_sh_dc_hi_bytes, lossless_rgb_png_bytes, ffv1.coded_payload_bytes, "sh_dc");
    print_codec_result("Lossy libx264rgb CRF28", &x264rgb_crf28, raw_sh_dc_hi_bytes, lossless_rgb_png_bytes, ffv1.coded_payload_bytes, "sh_dc");

    println!("Conclusion:");
    println!(
        "Lossless RGB PNG remains the direct image baseline at {} bytes.",
        lossless_rgb_png_bytes
    );
    println!(
        "Lossless FFV1 payload is {} bytes with max additional RMSE {:.9}.",
        ffv1.coded_payload_bytes,
        max_abs(&ffv1.additional_error)
    );
    println!(
        "Lossy libx264rgb CRF12 payload is {} bytes with max additional RMSE {:.9}.",
        x264rgb_crf12.coded_payload_bytes,
        max_abs(&x264rgb_crf12.additional_error)
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

    let best = most_promising_lossy([
        ("CRF12", &x264rgb_crf12),
        ("CRF18", &x264rgb_crf18),
        ("CRF28", &x264rgb_crf28),
    ]);
    println!(
        "Most promising operating point: {} with {} bytes and max additional RMSE {:.9}.",
        best.0,
        best.1.coded_payload_bytes,
        max_abs(&best.1.additional_error)
    );
    print_summary_block(
        "sh_dc_hi",
        &[
            ("Lossless FFV1", &ffv1, raw_sh_dc_hi_bytes),
            ("Lossy libx264rgb CRF12", &x264rgb_crf12, raw_sh_dc_hi_bytes),
            ("Lossy libx264rgb CRF18", &x264rgb_crf18, raw_sh_dc_hi_bytes),
            ("Lossy libx264rgb CRF28", &x264rgb_crf28, raw_sh_dc_hi_bytes),
        ],
        "sh_dc",
    );

    Ok(())
}

fn run_mixed_mode(input_ply: &str, container_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
    let scene = load_gaussians_from_ply(input_ply)?;
    let sorted = sort_gaussians_by_morton(&scene);
    let packed = read_packed_raster_scene_dir(container_dir)?;
    let baseline_scene = unpack_raster_to_scene(&packed);
    let baseline = compute_roundtrip_diagnostics(&sorted, &baseline_scene.gaussians, &packed);

    let work_dir = Path::new(container_dir).join("mixed_hi_policy_eval");
    std::fs::create_dir_all(&work_dir)?;

    let geom_png = write_geom_hi_rgb_png(&packed, work_dir.join("geom_hi_lossless_png"))?.geom_hi_rgb;
    let geom_hi_decoded = read_rgb_png_to_planes(&geom_png, packed.width, packed.height)?;
    let geom_bytes = std::fs::metadata(&geom_png)?.len();
    let raw_geom_hi_bytes = packed.geom_x.len() as u64 * 3;

    let sh_dc_png = write_sh_dc_hi_rgb_png(&packed, work_dir.join("sh_dc_hi_source"))?.sh_dc_hi_rgb;
    let sh_dc_codec = run_sh_dc_crf12_roundtrip(&sh_dc_png, &work_dir)?;
    let sh_dc_hi_decoded = read_rgb_png_to_planes(&sh_dc_codec.decoded_png, packed.width, packed.height)?;
    let raw_sh_dc_hi_bytes = packed.sh_dc_r.len() as u64 * 3;

    let with_geom = rebuild_packed_scene_with_decoded_geom_hi(
        &packed,
        [&geom_hi_decoded[0], &geom_hi_decoded[1], &geom_hi_decoded[2]],
    );
    let rebuilt = rebuild_packed_scene_with_decoded_sh_dc_hi(
        &with_geom,
        [&sh_dc_hi_decoded[0], &sh_dc_hi_decoded[1], &sh_dc_hi_decoded[2]],
    );
    let reconstructed = unpack_raster_to_scene(&rebuilt);

    let geometry_rmse = [
        rmse_for_xyz_axis(&sorted, &reconstructed.gaussians, 0),
        rmse_for_xyz_axis(&sorted, &reconstructed.gaussians, 1),
        rmse_for_xyz_axis(&sorted, &reconstructed.gaussians, 2),
    ];
    let geometry_additional = [
        geometry_rmse[0] - baseline.xyz_rmse[0],
        geometry_rmse[1] - baseline.xyz_rmse[1],
        geometry_rmse[2] - baseline.xyz_rmse[2],
    ];

    let sh_dc_rmse = [
        rmse_for_sh_dc_axis(&sorted, &reconstructed.gaussians, 0),
        rmse_for_sh_dc_axis(&sorted, &reconstructed.gaussians, 1),
        rmse_for_sh_dc_axis(&sorted, &reconstructed.gaussians, 2),
    ];
    let sh_dc_additional = [
        sh_dc_rmse[0] - baseline.sh_dc_rmse[0],
        sh_dc_rmse[1] - baseline.sh_dc_rmse[1],
        sh_dc_rmse[2] - baseline.sh_dc_rmse[2],
    ];

    let total_coded_bytes = geom_bytes + sh_dc_codec.coded_payload_bytes;

    println!("Input PLY: {}", input_ply);
    println!("Container directory: {}", container_dir);
    println!("Work directory: {}", work_dir.display());
    println!("Policy:");
    println!("  geom_hi_rgb -> lossless RGB PNG");
    println!("  sh_dc_hi_rgb -> lossy libx264rgb CRF12");
    println!("Bytes:");
    println!("  geom_hi lossless RGB PNG: {}", geom_bytes);
    println!("  sh_dc_hi lossy CRF12 payload: {}", sh_dc_codec.coded_payload_bytes);
    println!("  total coded bytes: {}", total_coded_bytes);
    println!("Geometry:");
    println!(
        "  RMSE: x={:.9}, y={:.9}, z={:.9}",
        geometry_rmse[0], geometry_rmse[1], geometry_rmse[2]
    );
    println!(
        "  additional RMSE beyond baseline: x={:.9}, y={:.9}, z={:.9}",
        geometry_additional[0], geometry_additional[1], geometry_additional[2]
    );
    println!("SH DC:");
    println!(
        "  RMSE: r={:.9}, g={:.9}, b={:.9}",
        sh_dc_rmse[0], sh_dc_rmse[1], sh_dc_rmse[2]
    );
    println!(
        "  additional RMSE beyond baseline: r={:.9}, g={:.9}, b={:.9}",
        sh_dc_additional[0], sh_dc_additional[1], sh_dc_additional[2]
    );
    println!("Artifacts:");
    println!("  geom_hi png: {}", geom_png.display());
    println!("  sh_dc_hi source png: {}", sh_dc_png.display());
    println!("  sh_dc_hi coded payload: {}", sh_dc_codec.coded_path.display());
    println!("  sh_dc_hi decoded png: {}", sh_dc_codec.decoded_png.display());
    print_mixed_summary_block(
        raw_geom_hi_bytes,
        raw_sh_dc_hi_bytes,
        geom_bytes,
        sh_dc_codec.coded_payload_bytes,
        total_coded_bytes,
        geometry_rmse,
        geometry_additional,
        [
            additional_ratio(geometry_additional[0], baseline.xyz_rmse[0]),
            additional_ratio(geometry_additional[1], baseline.xyz_rmse[1]),
            additional_ratio(geometry_additional[2], baseline.xyz_rmse[2]),
        ],
        [
            psnr_for_xyz_axis(&sorted, &reconstructed.gaussians, 0),
            psnr_for_xyz_axis(&sorted, &reconstructed.gaussians, 1),
            psnr_for_xyz_axis(&sorted, &reconstructed.gaussians, 2),
        ],
        [
            max_abs_xyz_axis(&sorted, &reconstructed.gaussians, 0),
            max_abs_xyz_axis(&sorted, &reconstructed.gaussians, 1),
            max_abs_xyz_axis(&sorted, &reconstructed.gaussians, 2),
        ],
        sh_dc_rmse,
        sh_dc_additional,
        [
            additional_ratio(sh_dc_additional[0], baseline.sh_dc_rmse[0]),
            additional_ratio(sh_dc_additional[1], baseline.sh_dc_rmse[1]),
            additional_ratio(sh_dc_additional[2], baseline.sh_dc_rmse[2]),
        ],
        [
            psnr_for_sh_dc_axis(&sorted, &reconstructed.gaussians, 0),
            psnr_for_sh_dc_axis(&sorted, &reconstructed.gaussians, 1),
            psnr_for_sh_dc_axis(&sorted, &reconstructed.gaussians, 2),
        ],
        [
            max_abs_sh_dc_axis(&sorted, &reconstructed.gaussians, 0),
            max_abs_sh_dc_axis(&sorted, &reconstructed.gaussians, 1),
            max_abs_sh_dc_axis(&sorted, &reconstructed.gaussians, 2),
        ],
        Some(compute_rgb_ssim(&sh_dc_png, &sh_dc_codec.decoded_png)?),
    );

    Ok(())
}

fn run_geom_codec_roundtrip(
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

    let mut encode_args = vec!["-y".to_string(), "-i".to_string(), source_png.display().to_string()];
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

    let decoded_hi = load_geom_hi_rgb_png(&decoded_dir, packed.width, packed.height)?;
    let rebuilt = rebuild_packed_scene_with_decoded_geom_hi(
        packed,
        [&decoded_hi[0], &decoded_hi[1], &decoded_hi[2]],
    );
    let reconstructed = unpack_raster_to_scene(&rebuilt);
    let rmse = [
        rmse_for_xyz_axis(sorted, &reconstructed.gaussians, 0),
        rmse_for_xyz_axis(sorted, &reconstructed.gaussians, 1),
        rmse_for_xyz_axis(sorted, &reconstructed.gaussians, 2),
    ];
    let coded_payload_bytes = std::fs::metadata(&coded_path)?.len();
    let additional_error = [
        rmse[0] - baseline_xyz_rmse[0],
        rmse[1] - baseline_xyz_rmse[1],
        rmse[2] - baseline_xyz_rmse[2],
    ];

    Ok(CodecRunResult {
        coded_path,
        decoded_png: decoded_png.clone(),
        coded_payload_bytes,
        rmse,
        additional_error,
        additional_ratio: [
            additional_ratio(additional_error[0], baseline_xyz_rmse[0]),
            additional_ratio(additional_error[1], baseline_xyz_rmse[1]),
            additional_ratio(additional_error[2], baseline_xyz_rmse[2]),
        ],
        psnr: [
            psnr_for_xyz_axis(sorted, &reconstructed.gaussians, 0),
            psnr_for_xyz_axis(sorted, &reconstructed.gaussians, 1),
            psnr_for_xyz_axis(sorted, &reconstructed.gaussians, 2),
        ],
        max_abs_error: [
            max_abs_xyz_axis(sorted, &reconstructed.gaussians, 0),
            max_abs_xyz_axis(sorted, &reconstructed.gaussians, 1),
            max_abs_xyz_axis(sorted, &reconstructed.gaussians, 2),
        ],
        ssim: Some(compute_rgb_ssim(source_png, &decoded_png)?),
    })
}

fn run_predictive_geom_codec_roundtrip(
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

    let mut encode_args = vec!["-y".to_string(), "-i".to_string(), source_png.display().to_string()];
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
    let rmse = [
        rmse_for_xyz_axis(sorted, &reconstructed.gaussians, 0),
        rmse_for_xyz_axis(sorted, &reconstructed.gaussians, 1),
        rmse_for_xyz_axis(sorted, &reconstructed.gaussians, 2),
    ];
    let additional_error = [
        rmse[0] - baseline_xyz_rmse[0],
        rmse[1] - baseline_xyz_rmse[1],
        rmse[2] - baseline_xyz_rmse[2],
    ];
    let coded_payload_bytes = std::fs::metadata(&coded_path)?.len();

    Ok(CodecRunResult {
        coded_path,
        decoded_png: decoded_png.clone(),
        coded_payload_bytes,
        rmse,
        additional_error,
        additional_ratio: [
            additional_ratio(additional_error[0], baseline_xyz_rmse[0]),
            additional_ratio(additional_error[1], baseline_xyz_rmse[1]),
            additional_ratio(additional_error[2], baseline_xyz_rmse[2]),
        ],
        psnr: [
            psnr_for_xyz_axis(sorted, &reconstructed.gaussians, 0),
            psnr_for_xyz_axis(sorted, &reconstructed.gaussians, 1),
            psnr_for_xyz_axis(sorted, &reconstructed.gaussians, 2),
        ],
        max_abs_error: [
            max_abs_xyz_axis(sorted, &reconstructed.gaussians, 0),
            max_abs_xyz_axis(sorted, &reconstructed.gaussians, 1),
            max_abs_xyz_axis(sorted, &reconstructed.gaussians, 2),
        ],
        ssim: Some(compute_rgb_ssim(source_png, &decoded_png)?),
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

    let mut encode_args =
        vec!["-y".to_string(), "-i".to_string(), artifacts.residual_png.display().to_string()];
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
    let rmse = [
        rmse_for_xyz_axis(sorted, &reconstructed.gaussians, 0),
        rmse_for_xyz_axis(sorted, &reconstructed.gaussians, 1),
        rmse_for_xyz_axis(sorted, &reconstructed.gaussians, 2),
    ];
    let additional_error = [
        rmse[0] - baseline_xyz_rmse[0],
        rmse[1] - baseline_xyz_rmse[1],
        rmse[2] - baseline_xyz_rmse[2],
    ];
    let coded_payload_bytes = std::fs::metadata(&coded_path)?.len() + artifacts.anchor_bytes() as u64;

    Ok(CodecRunResult {
        coded_path,
        decoded_png: decoded_png.clone(),
        coded_payload_bytes,
        rmse,
        additional_error,
        additional_ratio: [
            additional_ratio(additional_error[0], baseline_xyz_rmse[0]),
            additional_ratio(additional_error[1], baseline_xyz_rmse[1]),
            additional_ratio(additional_error[2], baseline_xyz_rmse[2]),
        ],
        psnr: [
            psnr_for_xyz_axis(sorted, &reconstructed.gaussians, 0),
            psnr_for_xyz_axis(sorted, &reconstructed.gaussians, 1),
            psnr_for_xyz_axis(sorted, &reconstructed.gaussians, 2),
        ],
        max_abs_error: [
            max_abs_xyz_axis(sorted, &reconstructed.gaussians, 0),
            max_abs_xyz_axis(sorted, &reconstructed.gaussians, 1),
            max_abs_xyz_axis(sorted, &reconstructed.gaussians, 2),
        ],
        ssim: Some(compute_rgb_ssim(&artifacts.residual_png, &decoded_png)?),
    })
}

fn run_sh_dc_codec_roundtrip(
    sorted: &[Gaussian],
    packed: &gaussian_packing::PackedRasterScene,
    baseline_sh_dc_rmse: &[f64; 3],
    work_dir: &Path,
    source_png: &Path,
    label: &'static str,
    codec_args: &[&str],
) -> Result<CodecRunResult, Box<dyn std::error::Error>> {
    let coded_path = work_dir.join(format!("{label}.mkv"));
    let decoded_dir = work_dir.join(format!("{label}_decoded"));
    std::fs::create_dir_all(&decoded_dir)?;
    let decoded_png = decoded_dir.join("sh_dc_hi_rgb.png");

    let mut encode_args = vec!["-y".to_string(), "-i".to_string(), source_png.display().to_string()];
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

    let decoded_hi = read_rgb_png_to_planes(&decoded_png, packed.width, packed.height)?;
    let rebuilt = rebuild_packed_scene_with_decoded_sh_dc_hi(
        packed,
        [&decoded_hi[0], &decoded_hi[1], &decoded_hi[2]],
    );
    let reconstructed = unpack_raster_to_scene(&rebuilt);

    let rmse = [
        rmse_for_sh_dc_axis(sorted, &reconstructed.gaussians, 0),
        rmse_for_sh_dc_axis(sorted, &reconstructed.gaussians, 1),
        rmse_for_sh_dc_axis(sorted, &reconstructed.gaussians, 2),
    ];
    let additional_error = [
        rmse[0] - baseline_sh_dc_rmse[0],
        rmse[1] - baseline_sh_dc_rmse[1],
        rmse[2] - baseline_sh_dc_rmse[2],
    ];
    let coded_payload_bytes = std::fs::metadata(&coded_path)?.len();

    Ok(CodecRunResult {
        coded_path,
        decoded_png: decoded_png.clone(),
        coded_payload_bytes,
        rmse,
        additional_error,
        additional_ratio: [
            additional_ratio(additional_error[0], baseline_sh_dc_rmse[0]),
            additional_ratio(additional_error[1], baseline_sh_dc_rmse[1]),
            additional_ratio(additional_error[2], baseline_sh_dc_rmse[2]),
        ],
        psnr: [
            psnr_for_sh_dc_axis(sorted, &reconstructed.gaussians, 0),
            psnr_for_sh_dc_axis(sorted, &reconstructed.gaussians, 1),
            psnr_for_sh_dc_axis(sorted, &reconstructed.gaussians, 2),
        ],
        max_abs_error: [
            max_abs_sh_dc_axis(sorted, &reconstructed.gaussians, 0),
            max_abs_sh_dc_axis(sorted, &reconstructed.gaussians, 1),
            max_abs_sh_dc_axis(sorted, &reconstructed.gaussians, 2),
        ],
        ssim: Some(compute_rgb_ssim(source_png, &decoded_png)?),
    })
}

fn run_sh_dc_crf12_roundtrip(
    source_png: &Path,
    work_dir: &Path,
) -> Result<ShDcCodecArtifacts, Box<dyn std::error::Error>> {
    let sh_dc_dir = work_dir.join("sh_dc_hi_crf12");
    std::fs::create_dir_all(&sh_dc_dir)?;
    let coded_path = sh_dc_dir.join("sh_dc_hi_rgb_crf12.mkv");
    let decoded_png = sh_dc_dir.join("sh_dc_hi_rgb.png");

    let encode_args = vec![
        "-y".to_string(),
        "-i".to_string(),
        source_png.display().to_string(),
        "-c:v".to_string(),
        "libx264rgb".to_string(),
        "-crf".to_string(),
        "12".to_string(),
        "-preset".to_string(),
        "medium".to_string(),
        "-pix_fmt".to_string(),
        "rgb24".to_string(),
        coded_path.display().to_string(),
    ];
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

    Ok(ShDcCodecArtifacts {
        coded_payload_bytes: std::fs::metadata(&coded_path)?.len(),
        coded_path,
        decoded_png,
    })
}

fn print_codec_result(
    title: &str,
    result: &CodecRunResult,
    raw_bytes: u64,
    lossless_rgb_png_bytes: u64,
    lossless_ffv1_bytes: u64,
    metric_label: &str,
) {
    println!("{title}:");
    println!("  coded payload: {}", result.coded_path.display());
    println!("  decoded png: {}", result.decoded_png.display());
    println!("  coded payload bytes: {}", result.coded_payload_bytes);
    println!("  coded/raw ratio: {:.6}", result.coded_payload_bytes as f64 / raw_bytes as f64);
    println!(
        "  coded/lossless-rgb-png ratio: {:.6}",
        result.coded_payload_bytes as f64 / lossless_rgb_png_bytes as f64
    );
    println!(
        "  coded/lossless-ffv1 ratio: {:.6}",
        result.coded_payload_bytes as f64 / lossless_ffv1_bytes as f64
    );
    match metric_label {
        "geometry" => {
            println!(
                "  geometry RMSE: x={:.9}, y={:.9}, z={:.9}",
                result.rmse[0], result.rmse[1], result.rmse[2]
            );
            println!(
                "  additional RMSE beyond baseline: x={:.9}, y={:.9}, z={:.9}",
                result.additional_error[0], result.additional_error[1], result.additional_error[2]
            );
        }
        "sh_dc" => {
            println!(
                "  SH DC RMSE: r={:.9}, g={:.9}, b={:.9}",
                result.rmse[0], result.rmse[1], result.rmse[2]
            );
            println!(
                "  additional RMSE beyond baseline: r={:.9}, g={:.9}, b={:.9}",
                result.additional_error[0], result.additional_error[1], result.additional_error[2]
            );
        }
        _ => {}
    }
}

fn print_summary_block(
    mode: &str,
    entries: &[(&str, &CodecRunResult, u64)],
    metric_label: &str,
) {
    println!("Summary Metrics [{}]:", mode);
    for (label, result, raw_bytes) in entries {
        let compression_percent = 100.0 * (1.0 - result.coded_payload_bytes as f64 / *raw_bytes as f64);
        let compression_ratio = *raw_bytes as f64 / result.coded_payload_bytes as f64;
        let ssim_text = result
            .ssim
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "n/a".to_string());
        let (a, b, c) = (result.rmse[0], result.rmse[1], result.rmse[2]);
        let (da, db, dc) = (
            result.additional_error[0],
            result.additional_error[1],
            result.additional_error[2],
        );
        let (ra, rb, rc) = (
            result.additional_ratio[0],
            result.additional_ratio[1],
            result.additional_ratio[2],
        );
        let (pa, pb, pc) = (format_f64(result.psnr[0]), format_f64(result.psnr[1]), format_f64(result.psnr[2]));
        let (ma, mb, mc) = (
            result.max_abs_error[0],
            result.max_abs_error[1],
            result.max_abs_error[2],
        );
        match metric_label {
            "geometry" | "sh_dc" => println!(
                "- {}: comp={:.2}% ratio={:.4} rmse=[{:.9},{:.9},{:.9}] addl=[{:.9},{:.9},{:.9}] addl/base=[{:.6},{:.6},{:.6}] psnr=[{},{},{}] maxabs=[{:.9},{:.9},{:.9}] ssim={}",
                label, compression_percent, compression_ratio, a, b, c, da, db, dc, ra, rb, rc, pa, pb, pc, ma, mb, mc, ssim_text
            ),
            _ => {}
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn print_mixed_summary_block(
    raw_geom_bytes: u64,
    raw_sh_dc_bytes: u64,
    geom_bytes: u64,
    sh_dc_bytes: u64,
    total_bytes: u64,
    geometry_rmse: [f64; 3],
    geometry_additional: [f64; 3],
    geometry_additional_ratio: [f64; 3],
    geometry_psnr: [f64; 3],
    geometry_max_abs: [f64; 3],
    sh_dc_rmse: [f64; 3],
    sh_dc_additional: [f64; 3],
    sh_dc_additional_ratio: [f64; 3],
    sh_dc_psnr: [f64; 3],
    sh_dc_max_abs: [f64; 3],
    sh_dc_ssim: Option<f64>,
) {
    let total_raw_bytes = raw_geom_bytes + raw_sh_dc_bytes;
    let total_compression_percent = 100.0 * (1.0 - total_bytes as f64 / total_raw_bytes as f64);
    let total_compression_ratio = total_raw_bytes as f64 / total_bytes as f64;
    println!("Summary Metrics [mixed]:");
    println!(
        "- total: raw_bytes={} coded_bytes={} geom_hi={} sh_dc_hi={} comp={:.2}% ratio={:.4}",
        total_raw_bytes, total_bytes, geom_bytes, sh_dc_bytes, total_compression_percent, total_compression_ratio
    );
    println!(
        "- geometry: rmse=[{:.9},{:.9},{:.9}] addl=[{:.9},{:.9},{:.9}] addl/base=[{:.6},{:.6},{:.6}] psnr=[{},{},{}] maxabs=[{:.9},{:.9},{:.9}] ssim=n/a",
        geometry_rmse[0], geometry_rmse[1], geometry_rmse[2],
        geometry_additional[0], geometry_additional[1], geometry_additional[2],
        geometry_additional_ratio[0], geometry_additional_ratio[1], geometry_additional_ratio[2],
        format_f64(geometry_psnr[0]), format_f64(geometry_psnr[1]), format_f64(geometry_psnr[2]),
        geometry_max_abs[0], geometry_max_abs[1], geometry_max_abs[2]
    );
    println!(
        "- sh_dc: rmse=[{:.9},{:.9},{:.9}] addl=[{:.9},{:.9},{:.9}] addl/base=[{:.6},{:.6},{:.6}] psnr=[{},{},{}] maxabs=[{:.9},{:.9},{:.9}] ssim={}",
        sh_dc_rmse[0], sh_dc_rmse[1], sh_dc_rmse[2],
        sh_dc_additional[0], sh_dc_additional[1], sh_dc_additional[2],
        sh_dc_additional_ratio[0], sh_dc_additional_ratio[1], sh_dc_additional_ratio[2],
        format_f64(sh_dc_psnr[0]), format_f64(sh_dc_psnr[1]), format_f64(sh_dc_psnr[2]),
        sh_dc_max_abs[0], sh_dc_max_abs[1], sh_dc_max_abs[2],
        sh_dc_ssim.map(|v| format!("{v:.6}")).unwrap_or_else(|| "n/a".to_string())
    );
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

fn rmse_for_xyz_axis(original: &[Gaussian], reconstructed: &[Gaussian], axis: usize) -> f64 {
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

fn max_abs_xyz_axis(original: &[Gaussian], reconstructed: &[Gaussian], axis: usize) -> f64 {
    original
        .iter()
        .zip(reconstructed.iter())
        .map(|(lhs, rhs)| (lhs.xyz[axis] as f64 - rhs.xyz[axis] as f64).abs())
        .fold(0.0_f64, f64::max)
}

fn psnr_for_xyz_axis(original: &[Gaussian], reconstructed: &[Gaussian], axis: usize) -> f64 {
    let rmse = rmse_for_xyz_axis(original, reconstructed, axis);
    let range = channel_range(original, |g| g.xyz[axis]);
    psnr_from_rmse(range, rmse)
}

fn rmse_for_sh_dc_axis(original: &[Gaussian], reconstructed: &[Gaussian], axis: usize) -> f64 {
    if original.is_empty() {
        return 0.0;
    }

    let mse = original
        .iter()
        .zip(reconstructed.iter())
        .map(|(lhs, rhs)| {
            let diff = lhs.sh_dc[axis] as f64 - rhs.sh_dc[axis] as f64;
            diff * diff
        })
        .sum::<f64>()
        / original.len() as f64;
    mse.sqrt()
}

fn max_abs_sh_dc_axis(original: &[Gaussian], reconstructed: &[Gaussian], axis: usize) -> f64 {
    original
        .iter()
        .zip(reconstructed.iter())
        .map(|(lhs, rhs)| (lhs.sh_dc[axis] as f64 - rhs.sh_dc[axis] as f64).abs())
        .fold(0.0_f64, f64::max)
}

fn psnr_for_sh_dc_axis(original: &[Gaussian], reconstructed: &[Gaussian], axis: usize) -> f64 {
    let rmse = rmse_for_sh_dc_axis(original, reconstructed, axis);
    let range = channel_range(original, |g| g.sh_dc[axis]);
    psnr_from_rmse(range, rmse)
}

fn channel_range(original: &[Gaussian], accessor: impl Fn(&Gaussian) -> f32) -> f64 {
    if original.is_empty() {
        return 0.0;
    }
    let mut min_v = accessor(&original[0]) as f64;
    let mut max_v = min_v;
    for g in original.iter().skip(1) {
        let v = accessor(g) as f64;
        min_v = min_v.min(v);
        max_v = max_v.max(v);
    }
    max_v - min_v
}

fn psnr_from_rmse(range: f64, rmse: f64) -> f64 {
    if rmse == 0.0 || range == 0.0 {
        f64::INFINITY
    } else {
        20.0 * (range / rmse).log10()
    }
}

fn additional_ratio(additional_rmse: f64, baseline_rmse: f64) -> f64 {
    if baseline_rmse.abs() < 1e-15 {
        if additional_rmse.abs() < 1e-15 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        additional_rmse / baseline_rmse
    }
}

fn max_abs(values: &[f64; 3]) -> f64 {
    values.iter().map(|value| value.abs()).fold(0.0_f64, f64::max)
}

fn compute_rgb_ssim(source_png: &Path, decoded_png: &Path) -> Result<f64, Box<dyn std::error::Error>> {
    let source = read_rgb_png_flexible(source_png)?;
    let decoded = read_rgb_png_flexible(decoded_png)?;
    Ok(rgb_ssim(&source, &decoded))
}

fn read_rgb_png_flexible(path: &Path) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let file = std::fs::File::open(path)?;
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info()?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let frame = reader.next_frame(&mut buf)?;
    Ok(buf[..frame.buffer_size()].to_vec())
}

fn rgb_ssim(source: &[u8], decoded: &[u8]) -> f64 {
    if source.len() != decoded.len() || source.is_empty() {
        return 0.0;
    }
    let mut acc = 0.0;
    for channel in 0..3 {
        let xs: Vec<f64> = source.iter().skip(channel).step_by(3).map(|&v| v as f64).collect();
        let ys: Vec<f64> = decoded.iter().skip(channel).step_by(3).map(|&v| v as f64).collect();
        acc += ssim_single_channel(&xs, &ys);
    }
    acc / 3.0
}

fn ssim_single_channel(xs: &[f64], ys: &[f64]) -> f64 {
    if xs.len() != ys.len() || xs.is_empty() {
        return 0.0;
    }
    let n = xs.len() as f64;
    let mean_x = xs.iter().sum::<f64>() / n;
    let mean_y = ys.iter().sum::<f64>() / n;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    let mut cov = 0.0;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        let dx = x - mean_x;
        let dy = y - mean_y;
        var_x += dx * dx;
        var_y += dy * dy;
        cov += dx * dy;
    }
    var_x /= n;
    var_y /= n;
    cov /= n;
    let l = 255.0;
    let c1 = (0.01 * l) * (0.01 * l);
    let c2 = (0.03 * l) * (0.03 * l);
    ((2.0 * mean_x * mean_y + c1) * (2.0 * cov + c2))
        / ((mean_x * mean_x + mean_y * mean_y + c1) * (var_x + var_y + c2))
}

fn format_f64(value: f64) -> String {
    if value.is_infinite() {
        "inf".to_string()
    } else {
        format!("{value:.6}")
    }
}

fn most_promising_lossy<'a>(
    candidates: [(&'a str, &'a CodecRunResult); 3],
) -> (&'a str, &'a CodecRunResult) {
    let mut best = candidates[0];
    for candidate in candidates.into_iter().skip(1) {
        let best_err = max_abs(&best.1.additional_error);
        let cand_err = max_abs(&candidate.1.additional_error);

        let better = if cand_err < best_err * 0.85 {
            true
        } else if cand_err <= best_err * 1.10 {
            candidate.1.coded_payload_bytes < best.1.coded_payload_bytes
        } else {
            false
        };

        if better {
            best = candidate;
        }
    }
    best
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
