use gaussian_parser::load_gaussians_from_ply;
use gaussian_sorter::generate_morton_code;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path : String = std::env::args().nth(1).expect("Missing .ply file path");

    let mut scene = load_gaussians_from_ply(&path)?;
    println!("Loaded {} Gaussians", scene.gaussians.len());
    
    scene.gaussians.sort_unstable_by_key(|g| generate_morton_code(g, &scene.mins, &scene.maxes));

    // Example: inspect the first Gaussian
    if let Some(g) = scene.gaussians.first() {
        println!("xyz={:?}, opacity={}, scale={:?}, rot={:?}", g.xyz, g.opacity, g.scale, g.rot);
        println!("sh_rest_len={}", g.sh_rest.len());
        println!("min xyz= {:?}", scene.mins);
        println!("max xyz= {:?}", scene.maxes);
    }
    Ok(())
}