use gaussian_parser::load_gaussians_from_ply;
use gaussian_packing::{
    compute_byte_split_plane_stats, compute_roundtrip_diagnostics, evaluate_geom_hi_png_roundtrip,
    evaluate_geom_hi_rgb_png_roundtrip, read_packed_raster_scene_dir, sort_gaussians_by_morton,
    unpack_raster_to_scene, write_geom_hi_pngs, write_geom_hi_rgb_png,
};

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

    let png_dir = std::path::Path::new(&container_dir).join("geom_hi_png_lossless");
    let png_paths = write_geom_hi_pngs(&packed, &png_dir)?;
    let gray_result = evaluate_geom_hi_png_roundtrip(&sorted, &packed, &png_dir)?;
    let rgb_dir = std::path::Path::new(&container_dir).join("geom_hi_rgb_png_lossless");
    let rgb_path = write_geom_hi_rgb_png(&packed, &rgb_dir)?;
    let rgb_result = evaluate_geom_hi_rgb_png_roundtrip(&sorted, &packed, &rgb_dir)?;
    let byte_stats = compute_byte_split_plane_stats(&packed);
    let added_error = [
        gray_result.xyz_rmse[0] - baseline.xyz_rmse[0],
        gray_result.xyz_rmse[1] - baseline.xyz_rmse[1],
        gray_result.xyz_rmse[2] - baseline.xyz_rmse[2],
    ];
    let rgb_added_error = [
        rgb_result.xyz_rmse[0] - baseline.xyz_rmse[0],
        rgb_result.xyz_rmse[1] - baseline.xyz_rmse[1],
        rgb_result.xyz_rmse[2] - baseline.xyz_rmse[2],
    ];

    println!("Input PLY: {}", input_ply);
    println!("Container directory: {}", container_dir);
    println!("PNG output directory: {}", png_dir.display());
    println!("Lossless geom hi PNG sizes:");
    println!("  geom_x_hi.png: {} bytes", gray_result.image_sizes[0]);
    println!("  geom_y_hi.png: {} bytes", gray_result.image_sizes[1]);
    println!("  geom_z_hi.png: {} bytes", gray_result.image_sizes[2]);
    println!("Total coded hi-byte grayscale PNG bytes: {}", gray_result.total_coded_bytes);
    println!("Lossless geom hi RGB PNG size:");
    println!("  geom_hi_rgb.png: {} bytes", rgb_result.total_coded_bytes);
    println!(
        "Geometry RMSE quantization-only baseline: x={:.9}, y={:.9}, z={:.9}",
        baseline.xyz_rmse[0], baseline.xyz_rmse[1], baseline.xyz_rmse[2]
    );
    println!(
        "Geometry RMSE after 3x grayscale PNG roundtrip: x={:.9}, y={:.9}, z={:.9}",
        gray_result.xyz_rmse[0], gray_result.xyz_rmse[1], gray_result.xyz_rmse[2]
    );
    println!(
        "Additional RMSE from 3x grayscale PNG coding: x={:.9}, y={:.9}, z={:.9}",
        added_error[0], added_error[1], added_error[2]
    );
    println!(
        "Geometry RMSE after RGB PNG roundtrip: x={:.9}, y={:.9}, z={:.9}",
        rgb_result.xyz_rmse[0], rgb_result.xyz_rmse[1], rgb_result.xyz_rmse[2]
    );
    println!(
        "Additional RMSE from RGB PNG coding: x={:.9}, y={:.9}, z={:.9}",
        rgb_added_error[0], rgb_added_error[1], rgb_added_error[2]
    );

    for name in ["geom_x", "geom_y", "geom_z"] {
        if let Some(entry) = byte_stats.iter().find(|entry| entry.name == name) {
            println!(
                "Geom hi-byte stats ({}): entropy_bits={:.6} delta_mean_abs={:.6} delta_std_abs={:.6} h_corr={:.6} v_corr={:.6}",
                name,
                entry.hi.entropy_bits_per_symbol,
                entry.hi.neighbor_delta.mean_abs,
                entry.hi.neighbor_delta.std_abs,
                entry.hi.horizontal_correlation.pearson_r,
                entry.hi.vertical_correlation.pearson_r
            );
            println!(
                "Geom lo-byte stats ({}): entropy_bits={:.6} delta_mean_abs={:.6} delta_std_abs={:.6} h_corr={:.6} v_corr={:.6}",
                name,
                entry.lo.entropy_bits_per_symbol,
                entry.lo.neighbor_delta.mean_abs,
                entry.lo.neighbor_delta.std_abs,
                entry.lo.horizontal_correlation.pearson_r,
                entry.lo.vertical_correlation.pearson_r
            );
        }
    }

    let raw_geom_hi_bytes = packed.geom_x.len() as u64 * 3;
    println!("Raw geom hi-byte payload bytes: {}", raw_geom_hi_bytes);
    println!(
        "3x grayscale PNG / raw geom hi-byte ratio: {:.6}",
        gray_result.total_coded_bytes as f64 / raw_geom_hi_bytes as f64
    );
    println!(
        "RGB PNG / raw geom hi-byte ratio: {:.6}",
        rgb_result.total_coded_bytes as f64 / raw_geom_hi_bytes as f64
    );
    println!(
        "Size comparison: 3x grayscale total={} bytes, 1x RGB={} bytes",
        gray_result.total_coded_bytes, rgb_result.total_coded_bytes
    );

    let max_added_error = added_error.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let max_rgb_added_error = rgb_added_error.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let gray_preserved_baseline = max_added_error <= 1e-12;
    let rgb_preserved_baseline = max_rgb_added_error <= 1e-12;
    if gray_preserved_baseline && rgb_preserved_baseline {
        if gray_result.total_coded_bytes < rgb_result.total_coded_bytes {
            println!(
                "Conclusion: 3 separate grayscale PNGs are more promising here. Both representations preserved the quantization-only baseline exactly, but grayscale coded smaller than RGB."
            );
        } else if rgb_result.total_coded_bytes < gray_result.total_coded_bytes {
            println!(
                "Conclusion: 1 combined RGB PNG is more promising here. Both representations preserved the quantization-only baseline exactly, and RGB coded smaller than the 3 grayscale images."
            );
        } else {
            println!(
                "Conclusion: both representations look equally promising here. They preserved the quantization-only baseline exactly and coded to the same size."
            );
        }
    } else if gray_preserved_baseline && !rgb_preserved_baseline {
        println!(
            "Conclusion: 3 separate grayscale PNGs are more promising here because they preserved the quantization-only baseline while the RGB PNG introduced extra error."
        );
    } else if !gray_preserved_baseline && rgb_preserved_baseline {
        println!(
            "Conclusion: 1 combined RGB PNG is more promising here because it preserved the quantization-only baseline while the grayscale PNG path introduced extra error."
        );
    } else {
        println!(
            "Conclusion: neither representation is promising yet because both introduced additional geometry error beyond the quantization-only baseline."
        );
    }

    println!(
        "Generated files: {}, {}, {}",
        png_paths.geom_x_hi.display(),
        png_paths.geom_y_hi.display(),
        png_paths.geom_z_hi.display()
    );
    println!("Generated file: {}", rgb_path.geom_hi_rgb.display());

    Ok(())
}
