use std::collections::HashMap;
use gaussian_parser::load_gaussians_from_ply;
use gaussian_sorter::generate_morton_code;
use rans_coding::{RansEnc, RansDec};
// use rand::Rng;
use deflate_coder::{compress_f32_vec, /*decompress_i32_vec*/};


fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path : String = std::env::args().nth(1).expect("Missing .ply file path");

    let mut scene = load_gaussians_from_ply(&path)?;
    // println!("Loaded {} Gaussians", scene.gaussians.len());

    scene.gaussians.sort_unstable_by_key(|g| generate_morton_code(g, &scene.mins, &scene.maxes));
    // //
    // println!("min xyz= {:?}", scene.mins);
    // println!("max xyz= {:?}", scene.maxes);
    //
    // for i in 0..1000 {
    //     if let Some(g) = scene.gaussians.get(i) {
    //         // println!("xyz={:?}, opacity={}, scale={:?}, rot={:?}", g.xyz, g.opacity, g.scale, g.rot);
            // println!("xyz={:?}", g.xyz);
            // println!("opacity={:?}", g.opacity);
            // println!("scale={:?}", g.scale);
            // println!("rotation={:?}", g.rot);
            // println!("sh first three={:?}, {:?}, {:?} ", g.sh_rest[0], g.sh_rest[1], g.sh_rest[2]);

            // println!("sh_rest_len={}", g.sh_rest.len());
        // }

    // }

    // let x1 = scene.gaussians.get(0).unwrap().xyz[0];
    // let x2 = scene.gaussians.get(1).unwrap().xyz[0];
    // println!("Float value x {:#b}", scene.gaussians.first().unwrap().xyz[0].to_bits());
    // let val: f32 = x2 - x1;
    // println!("Float value demo val {:#b}", val.to_bits());

    //text dynamic context update
    // let mut context = Context::new();
    //
    // for i in 0 .. 200 {
    //     // println!("For {} freq={} cum_freq={}", i, context.get_freq(i), context.get_cum_freq(i));
    //     assert_eq!(context.get_cum_freq(i), i as i32);
    //     assert_eq!(context.get_freq(i), 1i32);
    //     assert_eq!(context.get_symbol_from_cum_freq(i as i32), i as i32);
    // }
    //
    //
    // let rand_update = rand::rng().random_range(0..200);
    // println!("rand_update {}", rand_update);
    // context.increment_freq(rand_update);
    //
    // for i in 0 .. 200 {
    //     println!("Symbol from cum_freq={} symbol={}", i as i32, context.get_symbol_from_cum_freq(i as i32));
    //     if i <= rand_update {assert_eq!(context.get_cum_freq(i), i as i32)} else {assert_eq!(context.get_cum_freq(i), i as i32 + 1);}
    //     if i != rand_update {assert_eq!(context.get_freq(i), 1i32)} else {assert_eq!(context.get_freq(i), 2i32);}
    // }

    let bytes_per_i32 = std::mem::size_of::<i32>();
    // let total_bytes = 512 * 1024; // 512 KiB
    // let num_elements = scene.gaussians.len();
    let total_bytes = bytes_per_i32 * scene.gaussians.len();
    let num_elements = (1 << 16);
    // let num_elements = (1 << 14) - 10 ;
    // let num_elements = 10;
    // let num_elements = (1 << 14) - 250;
    // let num_elements = total_bytes / bytes_per_i32;

    // println!("Number of elements: {}", num_elements);

    // let mut rng = rand::thread_rng();
    // let mut data: Vec<i32> = Vec::with_capacity(num_elements);

    // for _ in 0..(num_elements) {
    //     let value = rng.gen_range(-5..=5);
    //     data.push(value);
    // }

    let mut data: Vec<f32> = Vec::with_capacity(num_elements);

    for i in 0..num_elements {
        let val = scene.gaussians.get(i).unwrap().xyz[0];
        data.push(val);
    }


    let mut encoder = RansEnc::new(total_bytes * 4, &data); //init to 1Mib double the input data size.

    // let (_,_, mantissas) = encoder.componentize_forward_pass();
    //
    // let mut values : HashMap<u32,u32> = HashMap::new();
    //
    // for mantissa in mantissas {
    //     print!("{:?},", mantissa);
    //     // if values.contains_key(&mantissa) {
        //     values.insert(mantissa, values.get(&mantissa).unwrap() + 1);
        // } else {
        //     values.insert(mantissa, 1);
        // }
    // }

    // let mut sort_values : Vec<_> = values.iter().collect();

    // sort_values.sort_by(|a, b| b.1.cmp(&a.1));
    // sort_values.truncate(1000);
    // sort_values.sort_by(|a, b| a.0.cmp(&b.0));
    //
    // for (i, val) in sort_values.iter().enumerate() {
    //     println!("{:?}: {}", val.0, val.1);
    // }


    let (mut code, raw_bytes, mut quantized) = encoder.encode_values();

    println!("Raw data size:                      {:?}", data.len() * bytes_per_i32);
    println!("Adaptive rANS coded data size:      {:?}", code.len() + raw_bytes.len());

    let deflate_code = compress_f32_vec(data.clone())?;
    println!("DEFLATE (LZ77 + Huffman) code size: {:?}\n", deflate_code.len());

    //
    let data_size  = (data.len() * bytes_per_i32) as f32;
    let code_size = code.len() as f32 + raw_bytes.len() as f32;
    let deflate_code_size = deflate_code.len() as f32;


    let coded_improvement = ((data_size - code_size) / data_size) * 100f32;
    let deflate_code_improvement = ((deflate_code_size - code_size) / deflate_code_size) * 100f32;
    //
    println!("Percent improvement my adaptive rANS data: {:?}", coded_improvement);
    println!("Percent improvement vs DEFLATE:            {:?}", deflate_code_improvement);
    //
    //
    let mut decoder = RansDec::new(code.as_mut_slice(), raw_bytes);
    // //
    let res = decoder.decode_values(num_elements);

    for i in 1 .. quantized.len() {
        quantized[i] = quantized[i] + quantized[i - 1];
    }

    // println!("{:?}", &data[ data.len() - 20.. data.len()]);
    // println!("{:?}", &res[res.len() - 20..res.len()]);
    // println!("Coded data: {:?}", code);
    println!("Quant data len: {}, res data len {}", quantized.len(), res.len());
    println!("Original data: {:?}", &data[data.len() - 10..]);
    println!("Original  quantized data: {:?}", &quantized[quantized.len() - 10..]);
     println!("Decoded quantized data: {:?}", &res[res.len() - 10..]);
    // println!("Original  quantized data: {:?}", &quantized[..10]);
    // println!("Decoded quantized data: {:?}", &res[..10]);
    // assert_eq!(quantized, res);


    Ok(())
}