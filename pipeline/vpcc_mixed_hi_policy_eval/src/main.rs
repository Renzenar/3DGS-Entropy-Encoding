use std::path::{Path, PathBuf};
use std::process::Command;

use gaussian_parser::load_gaussians_from_ply;
use gaussian_packing::{
    compute_roundtrip_diagnostics, read_packed_raster_scene_dir, read_rgb_png_to_planes,
    rebuild_packed_scene_with_decoded_geom_hi, rebuild_packed_scene_with_decoded_sh_dc_hi,
    sort_gaussians_by_morton, unpack_raster_to_scene, write_geom_hi_rgb_png, write_sh_dc_hi_rgb_png,
};
use gaussian_types::Gaussian;

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

    let work_dir = Path::new(&container_dir).join("mixed_hi_policy_eval");
    std::fs::create_dir_all(&work_dir)?;

    let geom_png = write_geom_hi_rgb_png(&packed, work_dir.join("geom_hi_lossless_png"))?.geom_hi_rgb;
    let geom_hi_decoded = read_rgb_png_to_planes(&geom_png, packed.width, packed.height)?;
    let geom_bytes = std::fs::metadata(&geom_png)?.len();

    let sh_dc_png = write_sh_dc_hi_rgb_png(&packed, work_dir.join("sh_dc_hi_source"))?.sh_dc_hi_rgb;
    let sh_dc_codec = run_sh_dc_crf12_roundtrip(&sh_dc_png, &work_dir)?;
    let sh_dc_hi_decoded = read_rgb_png_to_planes(&sh_dc_codec.decoded_png, packed.width, packed.height)?;

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

    Ok(())
}

struct ShDcCodecArtifacts {
    coded_path: PathBuf,
    decoded_png: PathBuf,
    coded_payload_bytes: u64,
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
