use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use gaussian_parser::load_gaussians_from_ply;
use gaussian_sorter::generate_morton_code;
use rans_coding::{RansEnc, RansDec};
use decoder::{EncodedAttribute, read_gaussian_from_gsz, write_gaussians_to_ply};
// use rand::Rng;
use deflate_coder::{compress_f32_vec, /*decompress_i32_vec*/};


fn encode_stream(data: &Vec<f32>) -> (Vec<u8>, Vec<u8>, Vec<f32>) {
    // rough capacity based on input size
    let bytes_per_i32 = std::mem::size_of::<i32>();
    let est_bytes = data.len() * bytes_per_i32;
    let mut encoder = RansEnc::new(est_bytes * 2, data);

    // rANS encode single integer stream
    encoder.encode_values()
}

// fn delta_encode(v: &mut [i32]) {
//     if v.len() < 2 { return; }
//     for i in (1..v.len()).rev() {
//         v[i] = v[i] - v[i-1];
//     }
// }
//
// fn delta_decode(v: &mut [i32]) {
//     if v.len() < 2 { return; }
//     for i in 1..v.len() {
//         v[i] = v[i] + v[i-1];
//     }
// }

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // load ply path from args
    let path: String = std::env::args().nth(1).expect("Missing .ply file path");
    let file_path: String = std::env::args().nth(2).expect("Missing .gsz file path");
    println!("Input PLY: {}", path);

    // load gaussian scene
    let mut scene = load_gaussians_from_ply(&path)?;

    let num_gaussians = scene.gaussians.len();
    println!("Loaded {} Gaussians", num_gaussians);

    // morton code generation
    scene.gaussians.sort_unstable_by_key(|g| generate_morton_code(g, &scene.mins, &scene.maxes));

    println!("min xyz = {:?}", scene.mins);
    println!("max xyz = {:?}", scene.maxes);

    // determine sh_rest length and sanity check
    let sh_rest_len = scene.gaussians[0].sh_rest.len();
    println!("sh_rest length per gaussian: {}", sh_rest_len);

    // position streams
    let mut s_xyz_x: Vec<f32> = Vec::with_capacity(num_gaussians);
    let mut s_xyz_y: Vec<f32> = Vec::with_capacity(num_gaussians);
    let mut s_xyz_z: Vec<f32> = Vec::with_capacity(num_gaussians);

    // normals streams (optional, presence + components)
    // let mut s_normals_present: Vec<i32> = Vec::with_capacity(num_gaussians);
    // let mut s_normals_x: Vec<i32> = Vec::with_capacity(num_gaussians);
    // let mut s_normals_y: Vec<i32> = Vec::with_capacity(num_gaussians);
    // let mut s_normals_z: Vec<i32> = Vec::with_capacity(num_gaussians);

    // spherical harmonics streams
    let mut s_sh_dc_r: Vec<f32> = Vec::with_capacity(num_gaussians);
    let mut s_sh_dc_g: Vec<f32> = Vec::with_capacity(num_gaussians);
    let mut s_sh_dc_b: Vec<f32> = Vec::with_capacity(num_gaussians);

    //sh_rest streams
    let mut s_sh_rest: Vec<Vec<f32>> = (0..sh_rest_len)
        .map(|_| Vec::with_capacity(num_gaussians))
        .collect();

    // opacity stream
    let mut s_opacity: Vec<f32> = Vec::with_capacity(num_gaussians);

    // scale streams
    let mut s_scale_x: Vec<f32> = Vec::with_capacity(num_gaussians);
    let mut s_scale_y: Vec<f32> = Vec::with_capacity(num_gaussians);
    let mut s_scale_z: Vec<f32> = Vec::with_capacity(num_gaussians);

    // rotation quaternion streams
    let mut s_rot_x: Vec<f32> = Vec::with_capacity(num_gaussians);
    let mut s_rot_y: Vec<f32> = Vec::with_capacity(num_gaussians);
    let mut s_rot_z: Vec<f32> = Vec::with_capacity(num_gaussians);
    let mut s_rot_w: Vec<f32> = Vec::with_capacity(num_gaussians);

    // convert all gaussian fields to i32 streams (float cast)
    for g in &scene.gaussians {
        // xyz -> three streams
        s_xyz_x.push(g.xyz[0]);
        s_xyz_y.push(g.xyz[1]);
        s_xyz_z.push(g.xyz[2]);

        // normals -> presence + three component streams
        // if let Some(n) = g.normals {
        //     s_normals_present.push(1);
        //     s_normals_x.push(n[0] as i32);
        //     s_normals_y.push(n[1] as i32);
        //     s_normals_z.push(n[2] as i32);
        // } else {
        //     s_normals_present.push(0);
        //     s_normals_x.push(0);
        //     s_normals_y.push(0);
        //     s_normals_z.push(0);
        // }

        // sh dc -> 3 streams
        s_sh_dc_r.push(g.sh_dc[0]);
        s_sh_dc_g.push(g.sh_dc[1]);
        s_sh_dc_b.push(g.sh_dc[2]);

        for (j, coeff) in g.sh_rest.iter().enumerate() {
            s_sh_rest[j].push(*coeff);
        }

        // opacity -> single stream
        s_opacity.push(g.opacity);

        // scale -> 3 streams (log-scales as-is)
        s_scale_x.push(g.scale[0]);
        s_scale_y.push(g.scale[1]);
        s_scale_z.push(g.scale[2]);

        // rotation quaternion -> 4 streams
        s_rot_x.push(g.rot[0]);
        s_rot_y.push(g.rot[1]);
        s_rot_z.push(g.rot[2]);
        s_rot_w.push(g.rot[3]);
    }

    // vector to store streams
    let mut streams: Vec<(String, Vec<f32>)> = Vec::new();

    // add position streams
    streams.push(("xyz_x".to_string(), s_xyz_x.clone()));
    streams.push(("xyz_y".to_string(), s_xyz_y.clone()));
    streams.push(("xyz_z".to_string(), s_xyz_z.clone()));

    // add spherical harmonics streams
    streams.push(("sh_dc_r".to_string(), s_sh_dc_r.clone()));
    streams.push(("sh_dc_g".to_string(), s_sh_dc_g.clone()));
    streams.push(("sh_dc_b".to_string(), s_sh_dc_b.clone()));


    //spherical harmonics rest streams
    for (j, s) in s_sh_rest.clone().into_iter().enumerate() {
        streams.push((format!("sh_rest_{j}"), s.clone()));
    }

    // add opacity stream
    streams.push(("opacity".to_string(), s_opacity.clone()));

    // add scale streams
    streams.push(("scale_x".to_string(), s_scale_x.clone()));
    streams.push(("scale_y".to_string(), s_scale_y.clone()));
    streams.push(("scale_z".to_string(), s_scale_z.clone()));

    // add rotation streams
    streams.push(("rot_x".to_string(), s_rot_x.clone()));
    streams.push(("rot_y".to_string(), s_rot_y.clone()));
    streams.push(("rot_z".to_string(), s_rot_z.clone()));
    streams.push(("rot_w".to_string(), s_rot_w.clone()));

    println!("Number of integer streams: {}", streams.len());

    // flatten all integer data for a deflate baseline
    let mut flat_data: Vec<f32> = Vec::new();
    for (_, s) in &streams {
        flat_data.extend_from_slice(s);
    }

    let bytes_per_i32 = std::mem::size_of::<i32>();

    let total_input_bytes = flat_data.len() * bytes_per_i32;

    println!("Total integer symbols: {}", flat_data.len());
    println!("Total raw bytes (i32): {}", total_input_bytes);

    // encode each stream with rANS and collect sizes
    let mut encoded: Vec<(&str, Vec<u8>, Vec<u8>, usize)> = Vec::new();
    let mut total_rans_bytes: usize = 0;

    let mut quantized : Vec<Vec<f32>> = Vec::new();

    for (name, original) in &streams {
        let mut data = original.clone();
        let (mut code, raw_bytes, mut quantized_bytes) = encode_stream(&data);

        for i in 1 .. quantized_bytes.len() {
                quantized_bytes[i] = quantized_bytes[i] + quantized_bytes[i - 1];
            };
        quantized.push(quantized_bytes);

        let stream_size = code.len() + raw_bytes.len();
        total_rans_bytes += stream_size;

        encoded.push((name, code, raw_bytes, data.len()));
    }

    ///write encoded data
    let file = File::create(file_path.clone())?;
    let mut writer = std::io::BufWriter::new(file);

    let scene_len = num_gaussians as u32;
    writer.write_all(&scene_len.to_le_bytes())?;

    for (name, code, raw_bytes, len) in encoded.iter() {
        //write code_len
        let code_len = code.len() as u32;
        assert!(code_len <= u32::MAX, "code length too large");
        writer.write_all(&code_len.to_le_bytes())?;

        //write code
        writer.write_all(&code)?;

        //write raw bytes len
        let raw_len = raw_bytes.len() as u32;
        assert!(raw_len <= u32::MAX, "raw bytes length too large");
        writer.write_all(&raw_len.to_le_bytes())?;

        //write raw bytes
        writer.write_all(&raw_bytes)?;
    }

    writer.flush()?;



    println!(
        "\nTotal rANS coded size (all streams): {} bytes",
        total_rans_bytes
    );


    println!("\nReading from file\n");
    let mut coded_data = Vec::<EncodedAttribute>::new();
    let mut scene_len = 0;
    read_gaussian_from_gsz(&file_path, &mut coded_data, &mut scene_len )?;

    println!("Number of Gaussians in Read back: {}", scene_len);



    // deflate baseline on flattened data
    // let deflate_code = compress_i32_vec(flat_data.clone())?;
    // let deflate_bytes = deflate_code.len();
    // println!("DEFLATE (LZ77 + Huffman) code size: {} bytes\n", deflate_bytes);
    let input_f = total_input_bytes as f32;
    let rans_f = total_rans_bytes as f32;
    // let deflate_f = deflate_bytes as f32;
    //
    let rans_vs_raw = (input_f - rans_f) / input_f * 100.0;
    // let rans_vs_deflate = (deflate_f - rans_f) / deflate_f * 100.0;

    println!(
        "Percent improvement rANS vs raw:      {:.2} %",
        rans_vs_raw
    );
    // println!(
    //     "Percent improvement rANS vs DEFLATE: {:.2} %",
    //     rans_vs_deflate
    // );

    let mut decoded_gaus: Vec<Vec<f32>> = Vec::new();

    decoded_gaus.push(s_xyz_x.clone());
    decoded_gaus.push(s_xyz_y);
    decoded_gaus.push(s_xyz_z);

    decoded_gaus.push(s_sh_dc_r);
    decoded_gaus.push(s_sh_dc_g);
    decoded_gaus.push(s_sh_dc_b);

    for (j, s) in s_sh_rest.into_iter().enumerate() {
        decoded_gaus.push(s);
    }
    decoded_gaus.push(s_opacity);
    decoded_gaus.push(s_scale_x);
    decoded_gaus.push(s_scale_y);
    decoded_gaus.push(s_scale_z);

    decoded_gaus.push(s_rot_x);
    decoded_gaus.push(s_rot_y);
    decoded_gaus.push(s_rot_z);
    decoded_gaus.push(s_rot_w);

    // decode each stream and verify
    for (idx, mut data) in coded_data.into_iter().enumerate() {
        // println!("{}", name);
        let mut decoder = RansDec::new(data.coded.as_mut_slice(), data.raw);
        let decoded = decoder.decode_values(scene_len as usize);

        if idx < 6  {
            decoded_gaus[idx] = decoded.clone();
        }

        let quantize = &quantized[idx];
        assert_eq!(decoded, *quantize);
        // println!("Decoded stream {}: {:?}", idx, &decoded[0..10]);
        // println!("Decoded stream {}: {:?}", idx, decoded.len());
        // println!("Quantized stream {}: {:?}", idx, &quantize[0..10]);
        if decoded != *quantize {
            println!("STREAM MISMATCH: {}", idx);
        }
    }





    let path = "./output/truck_decoded.ply";
    write_gaussians_to_ply(&path, &decoded_gaus, scene_len)?;


    println!("\n\n Total Drift");
    println!("Original Last 5: {:?}", &s_xyz_x[s_xyz_x.len()-5..]);
    println!("Decoded Last 5: {:?}", &decoded_gaus[0][decoded_gaus[0].len()-5..]);

    Ok(())
}
