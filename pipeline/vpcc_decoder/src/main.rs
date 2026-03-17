use gaussian_packing::{read_packed_raster_scene_dir, unpack_raster_to_scene, write_scene_to_ply};

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

    Ok(())
}
