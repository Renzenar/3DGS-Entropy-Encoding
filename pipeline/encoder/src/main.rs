use gaussian_parser::load_gaussians_from_ply;
use gaussian_sorter::generate_morton_code;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path : String = std::env::args().nth(1).expect("Missing .ply file path");

    let mut scene = load_gaussians_from_ply(&path)?;
    println!("Loaded {} Gaussians", scene.gaussians.len());
    
    scene.gaussians.sort_unstable_by_key(|g| generate_morton_code(g, &scene.mins, &scene.maxes));

    println!("min xyz= {:?}", scene.mins);
    println!("max xyz= {:?}", scene.maxes);

    for i in 0..1000 {
        if let Some(g) = scene.gaussians.get(i) {
            // println!("xyz={:?}, opacity={}, scale={:?}, rot={:?}", g.xyz, g.opacity, g.scale, g.rot);
            // println!("xyz={:?}", g.xyz);
            // println!("opacity={:?}", g.opacity);
            // println!("scale={:?}", g.scale);
            // println!("rotation={:?}", g.rot);
            // println!("sh first three={:?}, {:?}, {:?} ", g.sh_rest[0], g.sh_rest[1], g.sh_rest[2]);

            // println!("sh_rest_len={}", g.sh_rest.len());
        }

    }

    // let x1 = scene.gaussians.get(0).unwrap().xyz[0];
    // let x2 = scene.gaussians.get(1).unwrap().xyz[0];
    // println!("Float value x {:#b}", scene.gaussians.first().unwrap().xyz[0].to_bits());
    // let val: f32 = x2 - x1;
    // println!("Float value demo val {:#b}", val.to_bits());
    Ok(())
}