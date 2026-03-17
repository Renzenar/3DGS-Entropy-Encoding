use std::path::{Path, PathBuf};
use std::process::Command;

use gaussian_parser::load_gaussians_from_ply;
use gaussian_packing::{
    compute_roundtrip_diagnostics, read_packed_raster_scene_dir, read_rgb_png_to_planes,
    rebuild_packed_scene_with_decoded_sh_dc_hi, sort_gaussians_by_morton, unpack_raster_to_scene,
    write_sh_dc_hi_rgb_png,
};
use gaussian_types::Gaussian;

struct CodecRunResult {
    coded_path: PathBuf,
    decoded_png: PathBuf,
    coded_payload_bytes: u64,
    sh_dc_rmse: [f64; 3],
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

    let work_dir = Path::new(&container_dir).join("sh_dc_hi_codec_eval");
    std::fs::create_dir_all(&work_dir)?;

    let source_png = write_sh_dc_hi_rgb_png(&packed, &work_dir)?.sh_dc_hi_rgb;
    let lossless_rgb_png_bytes = std::fs::metadata(&source_png)?.len();
    let raw_sh_dc_hi_bytes = packed.sh_dc_r.len() as u64 * 3;

    let ffv1 = run_codec_roundtrip(
        &sorted,
        &packed,
        &baseline.sh_dc_rmse,
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
        &baseline.sh_dc_rmse,
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

    let x264rgb_crf12 = run_codec_roundtrip(
        &sorted,
        &packed,
        &baseline.sh_dc_rmse,
        &work_dir,
        &source_png,
        "libx264rgb_crf12",
        &[
            "-c:v",
            "libx264rgb",
            "-crf",
            "12",
            "-preset",
            "medium",
            "-pix_fmt",
            "rgb24",
        ],
    )?;

    let x264rgb_crf28 = run_codec_roundtrip(
        &sorted,
        &packed,
        &baseline.sh_dc_rmse,
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

    print_codec_result(
        "Lossless FFV1",
        &ffv1,
        raw_sh_dc_hi_bytes,
        lossless_rgb_png_bytes,
        ffv1.coded_payload_bytes,
    );
    print_codec_result(
        "Lossy libx264rgb CRF12",
        &x264rgb_crf12,
        raw_sh_dc_hi_bytes,
        lossless_rgb_png_bytes,
        ffv1.coded_payload_bytes,
    );
    print_codec_result(
        "Lossy libx264rgb CRF18",
        &x264rgb_crf18,
        raw_sh_dc_hi_bytes,
        lossless_rgb_png_bytes,
        ffv1.coded_payload_bytes,
    );
    print_codec_result(
        "Lossy libx264rgb CRF28",
        &x264rgb_crf28,
        raw_sh_dc_hi_bytes,
        lossless_rgb_png_bytes,
        ffv1.coded_payload_bytes,
    );

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

    Ok(())
}

fn run_codec_roundtrip(
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

    let decoded_hi = read_rgb_png_to_planes(&decoded_png, packed.width, packed.height)?;
    let rebuilt = rebuild_packed_scene_with_decoded_sh_dc_hi(
        packed,
        [&decoded_hi[0], &decoded_hi[1], &decoded_hi[2]],
    );
    let reconstructed = unpack_raster_to_scene(&rebuilt);

    let sh_dc_rmse = [
        rmse_for_sh_dc_axis(sorted, &reconstructed.gaussians, 0),
        rmse_for_sh_dc_axis(sorted, &reconstructed.gaussians, 1),
        rmse_for_sh_dc_axis(sorted, &reconstructed.gaussians, 2),
    ];
    let additional_error = [
        sh_dc_rmse[0] - baseline_sh_dc_rmse[0],
        sh_dc_rmse[1] - baseline_sh_dc_rmse[1],
        sh_dc_rmse[2] - baseline_sh_dc_rmse[2],
    ];
    let coded_payload_bytes = std::fs::metadata(&coded_path)?.len();

    Ok(CodecRunResult {
        coded_path,
        decoded_png,
        coded_payload_bytes,
        sh_dc_rmse,
        additional_error,
    })
}

fn print_codec_result(
    title: &str,
    result: &CodecRunResult,
    raw_sh_dc_hi_bytes: u64,
    lossless_rgb_png_bytes: u64,
    lossless_ffv1_bytes: u64,
) {
    println!("{title}:");
    println!("  coded payload: {}", result.coded_path.display());
    println!("  decoded png: {}", result.decoded_png.display());
    println!("  coded payload bytes: {}", result.coded_payload_bytes);
    println!(
        "  coded/raw ratio: {:.6}",
        result.coded_payload_bytes as f64 / raw_sh_dc_hi_bytes as f64
    );
    println!(
        "  coded/lossless-rgb-png ratio: {:.6}",
        result.coded_payload_bytes as f64 / lossless_rgb_png_bytes as f64
    );
    println!(
        "  coded/lossless-ffv1 ratio: {:.6}",
        result.coded_payload_bytes as f64 / lossless_ffv1_bytes as f64
    );
    println!(
        "  SH DC RMSE: r={:.9}, g={:.9}, b={:.9}",
        result.sh_dc_rmse[0], result.sh_dc_rmse[1], result.sh_dc_rmse[2]
    );
    println!(
        "  additional RMSE beyond baseline: r={:.9}, g={:.9}, b={:.9}",
        result.additional_error[0], result.additional_error[1], result.additional_error[2]
    );
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

fn max_abs(values: &[f64; 3]) -> f64 {
    values.iter().map(|value| value.abs()).fold(0.0_f64, f64::max)
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
