use gaussian_parser::load_gaussians_from_ply;
use gaussian_packing::{
    compute_roundtrip_diagnostics, packed_raster_raw_payload_bytes, pack_scene_to_raster,
    sort_gaussians_by_morton, unpack_raster_to_scene, write_byte_split_planes_dir,
    write_debug_preview_pngs, write_packed_raster_scene_dir,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input_path = std::env::args().nth(1).expect("Missing input .ply path");
    let output_dir = std::env::args().nth(2).expect("Missing output directory path");
    let width = std::env::args()
        .nth(3)
        .map(|v| v.parse::<u32>())
        .transpose()?
        .unwrap_or(1024);
    let compare_gsz = std::env::args().nth(4);

    let scene = load_gaussians_from_ply(&input_path)?;
    let sorted = sort_gaussians_by_morton(&scene);
    let packed = pack_scene_to_raster(&scene, width);
    let reconstructed = unpack_raster_to_scene(&packed);
    let diagnostics = compute_roundtrip_diagnostics(&sorted, &reconstructed.gaussians, &packed);

    write_packed_raster_scene_dir(&packed, &output_dir)?;
    if std::env::var("VPCC_EXPORT_BYTE_SPLIT").ok().as_deref() == Some("1") {
        write_byte_split_planes_dir(&packed, &output_dir)?;
    }
    if std::env::var("VPCC_WRITE_PREVIEWS").ok().as_deref() == Some("1") {
        write_debug_preview_pngs(&packed, &output_dir)?;
    }

    println!("Input PLY: {}", input_path);
    println!("Output directory: {}", output_dir);
    println!("Atlas dimensions: {} x {}", packed.width, packed.height);
    println!("Gaussians packed: {}", packed.num_gaussians);
    let raw_payload_bytes = packed_raster_raw_payload_bytes(&packed);
    println!("Total raw payload bytes: {}", raw_payload_bytes);
    let original_ply_size = std::fs::metadata(&input_path)?.len();
    println!("Original PLY size: {}", original_ply_size);
    println!(
        "Raw payload / original PLY: {:.6}",
        raw_payload_bytes as f64 / original_ply_size as f64
    );
    if let Some(gsz_path) = compare_gsz {
        match std::fs::metadata(&gsz_path) {
            Ok(meta) => {
                let gsz_size = meta.len();
                println!("Comparison .gsz size: {}", gsz_size);
                println!(
                    "Raw payload / .gsz: {:.6}",
                    raw_payload_bytes as f64 / gsz_size as f64
                );
            }
            Err(err) => {
                println!("Comparison .gsz unavailable ({}): {}", gsz_path, err);
            }
        }
    }
    println!(
        "XYZ RMSE: x={:.9}, y={:.9}, z={:.9}",
        diagnostics.xyz_rmse[0], diagnostics.xyz_rmse[1], diagnostics.xyz_rmse[2]
    );
    println!(
        "SH_DC RMSE: r={:.9}, g={:.9}, b={:.9}",
        diagnostics.sh_dc_rmse[0], diagnostics.sh_dc_rmse[1], diagnostics.sh_dc_rmse[2]
    );
    for (label, channel) in [
        ("xyz_x", &diagnostics.original_neighbor_deltas.xyz[0]),
        ("xyz_y", &diagnostics.original_neighbor_deltas.xyz[1]),
        ("xyz_z", &diagnostics.original_neighbor_deltas.xyz[2]),
        ("sh_dc_r", &diagnostics.original_neighbor_deltas.sh_dc[0]),
        ("sh_dc_g", &diagnostics.original_neighbor_deltas.sh_dc[1]),
        ("sh_dc_b", &diagnostics.original_neighbor_deltas.sh_dc[2]),
    ] {
        println!(
            "Neighbor delta stats ({label}): count={} mean_abs={:.9} std_abs={:.9}",
            channel.count, channel.mean_abs, channel.std_abs
        );
    }
    for plane in &diagnostics.plane_stats {
        println!(
            "Plane stats ({}): min={} max={} entropy_bits={:.6} delta_mean_abs={:.6} delta_std_abs={:.6} h_corr={:.6} (n={}) v_corr={:.6} (n={})",
            plane.name,
            plane.min,
            plane.max,
            plane.entropy_bits_per_symbol,
            plane.neighbor_delta.mean_abs,
            plane.neighbor_delta.std_abs,
            plane.horizontal_correlation.pearson_r,
            plane.horizontal_correlation.count,
            plane.vertical_correlation.pearson_r,
            plane.vertical_correlation.count
        );
    }
    for entry in &diagnostics.byte_split_plane_stats {
        println!(
            "Byte-split summary ({}): u16_entropy={:.6} hi_entropy={:.6} lo_entropy={:.6} | u16_hcorr={:.6} hi_hcorr={:.6} lo_hcorr={:.6} | u16_vcorr={:.6} hi_vcorr={:.6} lo_vcorr={:.6}",
            entry.name,
            entry.base_u16.entropy_bits_per_symbol,
            entry.hi.entropy_bits_per_symbol,
            entry.lo.entropy_bits_per_symbol,
            entry.base_u16.horizontal_correlation.pearson_r,
            entry.hi.horizontal_correlation.pearson_r,
            entry.lo.horizontal_correlation.pearson_r,
            entry.base_u16.vertical_correlation.pearson_r,
            entry.hi.vertical_correlation.pearson_r,
            entry.lo.vertical_correlation.pearson_r
        );
        println!(
            "Byte-split detail ({}): hi[min={}, max={}, delta_mean_abs={:.6}, delta_std_abs={:.6}] lo[min={}, max={}, delta_mean_abs={:.6}, delta_std_abs={:.6}]",
            entry.name,
            entry.hi.min,
            entry.hi.max,
            entry.hi.neighbor_delta.mean_abs,
            entry.hi.neighbor_delta.std_abs,
            entry.lo.min,
            entry.lo.max,
            entry.lo.neighbor_delta.mean_abs,
            entry.lo.neighbor_delta.std_abs
        );
    }

    Ok(())
}
