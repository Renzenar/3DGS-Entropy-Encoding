use gaussian_parser::load_gaussians_from_ply;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path : String = std::env::args().nth(1).expect("Missing .ply file path");

    let scene = load_gaussians_from_ply(&path)?;
    println!("Loaded {} Gaussians", scene.gaussians.len());

    // Example: inspect the first Gaussian
    if let Some(g) = scene.gaussians.last() {
        println!("xyz={:?}, opacity={}, scale={:?}, rot={:?}", g.xyz, g.opacity, g.scale, g.rot);
        println!("sh_rest_len={}", g.sh_rest.len());
    }
    Ok(())
}