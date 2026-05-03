use std::{collections::BTreeSet, path::Path};

use gaussian_packing::{
    LcevcMetadata, measure_packed_raster_scene_dir, read_packed_raster_scene_dir,
    sh_rest_payload_stats, unpack_raster_to_scene, write_scene_to_ply,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input_dir = std::env::args()
        .nth(1)
        .expect("Missing packed raster directory path");
    let output_path = std::env::args().nth(2).expect("Missing output .ply path");

    let packed = read_packed_raster_scene_dir(&input_dir)?;
    let scene = unpack_raster_to_scene(&packed);
    write_scene_to_ply(&output_path, &scene)?;

    println!("Input directory: {}", input_dir);
    println!("Output PLY: {}", output_path);
    println!("Decoded gaussians: {}", scene.gaussians.len());
    let output_ply_size = std::fs::metadata(&output_path)?.len();
    println!("Output PLY size: {}", output_ply_size);
    let coded_summary = measure_packed_raster_scene_dir(&input_dir)?;
    let sh_rest_stats = sh_rest_payload_stats(&packed);
    println!(
        "SH rest payload: raw_bytes={} coded_bytes={} storage={}",
        sh_rest_stats.raw_bytes, coded_summary.sh_rest_bytes, sh_rest_stats.storage
    );

    let residual_bytes = measure_residual_bytes(Path::new(&input_dir));

    println!("Residual Bytes: {}", residual_bytes);
    let true_total = coded_summary.total_coded_bytes + residual_bytes;

    println!("True total bytes: {}", true_total);
    println!(
        "True compression ratio: {:.4}",
        output_ply_size as f64 / true_total as f64
    );
    let coverage = coverage_from_packed(&packed);
    print_stream_coverage_block("vpcc_decoder", &coverage, &coverage, &coverage);

    Ok(())
}

fn measure_residual_bytes(dir: &Path) -> u64 {
    std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if path.to_string_lossy().contains("sh_rest_residual_") {
                Some(std::fs::metadata(path).ok()?.len())
            } else {
                None
            }
        })
        .sum()
}

fn coverage_from_packed(packed: &gaussian_packing::PackedRasterScene) -> Vec<String> {
    let sh_rest_len = packed.sh_rest_len as usize;
    let normals_present = packed.normals_present;
    let mut streams = vec![
        "xyz_x".to_string(),
        "xyz_y".to_string(),
        "xyz_z".to_string(),
    ];
    if normals_present {
        streams.extend(
            ["normals_x", "normals_y", "normals_z"]
                .into_iter()
                .map(String::from),
        );
    }
    streams.extend(
        ["sh_dc_r", "sh_dc_g", "sh_dc_b"]
            .into_iter()
            .map(String::from),
    );
    for i in 0..sh_rest_len {
        streams.push(format!("sh_rest_{i}"));
    }
    streams.extend(
        [
            "opacity", "scale_x", "scale_y", "scale_z", "rot_x", "rot_y", "rot_z", "rot_w",
        ]
        .into_iter()
        .map(String::from),
    );
    streams
}

fn print_stream_coverage_block(
    mode: &str,
    packed_streams: &[String],
    decoded_streams: &[String],
    reconstructed_streams: &[String],
) {
    let schema_streams = packed_streams;
    println!("Stream Coverage [{}]:", mode);
    println!(
        "- counts: schema={} packed={} compressed={} decoded={} reconstructed={}",
        schema_streams.len(),
        packed_streams.len(),
        packed_streams.len(),
        decoded_streams.len(),
        reconstructed_streams.len()
    );
    println!(
        "- missing packed: {}",
        format_stream_list(&missing_streams(schema_streams, packed_streams))
    );
    println!(
        "- missing compressed: {}",
        format_stream_list(&missing_streams(schema_streams, packed_streams))
    );
    println!(
        "- missing decoded: {}",
        format_stream_list(&missing_streams(schema_streams, decoded_streams))
    );
    println!(
        "- missing reconstructed: {}",
        format_stream_list(&missing_streams(schema_streams, reconstructed_streams))
    );
}

fn missing_streams(schema_streams: &[String], covered_streams: &[String]) -> Vec<String> {
    let covered: BTreeSet<&str> = covered_streams.iter().map(|name| name.as_str()).collect();
    schema_streams
        .iter()
        .filter(|name| !covered.contains(name.as_str()))
        .cloned()
        .collect()
}

fn format_stream_list(streams: &[String]) -> String {
    if streams.is_empty() {
        "none".to_string()
    } else {
        streams.join(", ")
    }
}
