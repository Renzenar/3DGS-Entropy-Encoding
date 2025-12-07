use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use gaussian_parser::load_gaussians_from_ply;
use gaussian_sorter::generate_morton_code;
use rans_coding::{RansEnc, RansDec};
use decoder::{EncodedAttribute, read_gaussian_from_gsz, write_gaussians_to_ply};
// use rand::Rng;
use deflate_coder::{compress_f32_vec, /*decompress_i32_vec*/};
use gaussian_types::Gaussian;


fn encode_stream(data: &Vec<f32>) -> (Vec<u8>, Vec<u8>, Vec<f32>) {
    // rough capacity based on input size
    let bytes_per_i32 = std::mem::size_of::<i32>();
    let est_bytes = data.len() * bytes_per_i32;
    let mut encoder = RansEnc::new(est_bytes * 2, data);

    // rANS encode single integer stream
    encoder.encode_values()
}

fn delta_encode(v: &mut [f32]) {
    if v.len() < 2 { return; }
    for i in (1..v.len()).rev() {
        v[i] = v[i] - v[i-1];
    }
}

fn delta_decode(v: &mut [f32]) {
    if v.len() < 2 { return; }
    for i in 1..v.len() {
        v[i] = v[i] + v[i-1];
    }
}

//CHAT WROTE THESE BELOW

/// Convert a unit (or nearly-unit) quaternion (w, x, y, z) to a 3x3 rotation matrix.
/// Returns R such that v_world = R * v_local.
fn quat_to_mat3(w: f32, x: f32, y: f32, z: f32) -> [[f32; 3]; 3] {
    // Normalize in case it's slightly off unit length
    let norm2 = w*w + x*x + y*y + z*z;
    let (w, x, y, z) = if norm2 > 0.0 {
        let inv_norm = 1.0 / norm2.sqrt();
        (w * inv_norm, x * inv_norm, y * inv_norm, z * inv_norm)
    } else {
        // Degenerate case: identity rotation
        return [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
    };

    [
        [
            1.0 - 2.0 * (y*y + z*z),
            2.0 * (x*y - z*w),
            2.0 * (x*z + y*w),
        ],
        [
            2.0 * (x*y + z*w),
            1.0 - 2.0 * (x*x + z*z),
            2.0 * (y*z - x*w),
        ],
        [
            2.0 * (x*z - y*w),
            2.0 * (y*z + x*w),
            1.0 - 2.0 * (x*x + y*y),
        ],
    ]
}

/// Given:
/// - rotation quaternion in (w, x, y, z) order
/// - 3DGS log-scale vector (scale_x, scale_y, scale_z)
/// returns the world-axis standard deviations (σ_world_x, σ_world_y, σ_world_z).
///
/// This uses Σ = R * diag(σ^2) * R^T and then
/// var_j = Σ_jj, σ_world_j = sqrt(var_j).
pub fn world_axis_scales_from_quat_and_log_scale(
    rotation: [f32; 4],   // [w, x, y, z]
    log_scale: [f32; 3],  // stored 3DGS scale
) -> [f32; 3] {
    let [w, x, y, z] = rotation;
    let r = quat_to_mat3(w, x, y, z);

    // Local standard deviations σ = exp(scale)
    let sigma = [
        log_scale[0].exp(),
        log_scale[1].exp(),
        log_scale[2].exp(),
    ];

    let sigma2 = [
        sigma[0] * sigma[0],
        sigma[1] * sigma[1],
        sigma[2] * sigma[2],
    ];

    // For each world axis j, var_j = sum_k (R_jk^2 * σ_k^2)
    let mut world_sigma = [0.0f32; 3];

    for j in 0..3 {
        let rj0 = r[j][0];
        let rj1 = r[j][1];
        let rj2 = r[j][2];

        let var_j =
            rj0 * rj0 * sigma2[0] +
                rj1 * rj1 * sigma2[1] +
                rj2 * rj2 * sigma2[2];

        world_sigma[j] = var_j.max(0.0).sqrt(); // guard against tiny negative due to FP
    }

    world_sigma
}
///Position Prediction Function
//x_hat_i = x_{i-1} + (x_{i-1} - x_{i-2]) * (s_i / s_{i-1});
fn pred_func(axis: usize, idx: usize, gs: &Vec<Gaussian>) -> f32{
    let world_scale_cur = world_axis_scales_from_quat_and_log_scale(
        gs[idx].rot,
        gs[idx].scale,
    );
    let world_scale_prev = world_axis_scales_from_quat_and_log_scale(
        gs[idx - 1].rot,
        gs[idx - 1].scale,
    );
    let scale_rat = world_scale_cur[axis] / world_scale_prev[axis];
    let pred = gs[idx -1].xyz[axis] +(gs[idx - 1].xyz[axis] - gs[idx - 2].xyz[axis]) * scale_rat;

    pred
}

fn delta_encode_pos(gs: &mut Vec<Gaussian>) {
    if gs.len() < 3 { return; }
    for i in (2..gs.len()).rev() {
        let pred_x = get_pos_pred(0, i, gs);
        gs[i].xyz[0] = gs[i].xyz[0] - pred_x;
        let pred_y = get_pos_pred(1, i, gs);
        gs[i].xyz[1] = gs[i].xyz[1] - pred_y;
        let pred_z = get_pos_pred(2, i, gs);
        gs[i].xyz[2] = gs[i].xyz[2] - pred_z;
    }
}

//TODO decode function

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
        delta_encode(&mut data);
        println!("Delta coding complete: {:?}", &data[0..10]);
        let (mut code, raw_bytes, quantized_bytes) = encode_stream(&data);

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
    // let input_f = total_input_bytes as f32;
    // let rans_f = total_rans_bytes as f32;
    // let deflate_f = deflate_bytes as f32;

    let input_f = ((s_xyz_x.len() + s_xyz_y.len() + s_xyz_z.len()) * size_of::<f32>()) as f32;
    let rans_f = (encoded[0].1.len() + encoded[0].2.len() +
        encoded[1].1.len() + encoded[1].2.len() +
        encoded[2].1.len() + encoded[2].2.len())
        as f32;
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
        let mut decoded = decoder.decode_values(scene_len as usize);
        delta_decode(&mut decoded);

        if idx < 3  {
            decoded_gaus[idx] = decoded.clone();
        }

        let mut quantize = quantized[idx].clone();
        println!("Decoded stream {}: {:?}", idx, &decoded[0..10]);
        delta_decode(&mut quantize);
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
    // println!("Original First 5: {:?}", &s_xyz_x[0..5]);
    println!("Decoded Last 5: {:?}", &decoded_gaus[0][decoded_gaus[0].len()-5..]);
    // println!("Decoded First 5: {:?}", &decoded_gaus[0][0..5]);


    Ok(())
}
