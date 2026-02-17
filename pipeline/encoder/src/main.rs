use std::fs::File;
use std::io::Write;
use gaussian_parser::load_gaussians_from_ply;
use gaussian_sorter::generate_morton_code;
use rans_coding::RansEnc;
use indicatif::{ProgressBar, ProgressStyle};


fn encode_stream(data: &Vec<f32>, step: f32, err : i32, mant_scale: u32) -> (Vec<u8>, (Vec<u8>, u32) ) {
    // rough capacity based on input size
    let bytes_per_i32 = std::mem::size_of::<i32>();
    let est_bytes = data.len() * bytes_per_i32;
    let mut encoder = RansEnc::new(est_bytes * 2, data, step, err, mant_scale);

    encoder.debug_stats();
    // rANS encode single integer stream
    encoder.encode_values()
}


pub fn mean_std_u32(values: &Vec<u32>) -> (f64, f64) {
    let n = values.len();
    if n == 0 {
        return (0.0, 0.0);
    }

    // Compute mean
    let sum: f64 = values.iter().map(|&v| v as f64).sum();
    let mean = sum / n as f64;

    // Compute variance
    let var_sum: f64 = values.iter()
        .map(|&v| {
            let diff = v as f64 - mean;
            diff * diff
        })
        .sum();

    let variance = var_sum / n as f64; // population variance
    let std_dev = variance.sqrt();

    (mean, std_dev)
}

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

    // determine sh_rest length and sanity check
    let sh_rest_len = scene.gaussians[0].sh_rest.len();

    // position streams
    let mut s_xyz_x: Vec<f32> = Vec::with_capacity(num_gaussians);
    let mut s_xyz_y: Vec<f32> = Vec::with_capacity(num_gaussians);
    let mut s_xyz_z: Vec<f32> = Vec::with_capacity(num_gaussians);

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
    streams.push(("xyz_x".to_string(), s_xyz_x));
    streams.push(("xyz_y".to_string(), s_xyz_y));
    streams.push(("xyz_z".to_string(), s_xyz_z));

    // add spherical harmonics streams
    streams.push(("sh_dc_r".to_string(), s_sh_dc_r));
    streams.push(("sh_dc_g".to_string(), s_sh_dc_g));
    streams.push(("sh_dc_b".to_string(), s_sh_dc_b));


    //spherical harmonics rest streams
    for (j, s) in s_sh_rest.clone().into_iter().enumerate() {
        streams.push((format!("sh_rest_{j}"), s));
    }

    // add opacity stream
    streams.push(("opacity".to_string(), s_opacity));

    // add scale streams
    streams.push(("scale_x".to_string(), s_scale_x));
    streams.push(("scale_y".to_string(), s_scale_y));
    streams.push(("scale_z".to_string(), s_scale_z));

    // add rotation streams
    streams.push(("rot_x".to_string(), s_rot_x));
    streams.push(("rot_y".to_string(), s_rot_y));
    streams.push(("rot_z".to_string(), s_rot_z));
    streams.push(("rot_w".to_string(), s_rot_w));

    println!("Number of integer streams: {}", streams.len());

    // flatten all integer data for a deflate baseline
    let mut flat_data: Vec<f32> = Vec::new();
    for (_, s) in &streams {
        flat_data.extend_from_slice(s);
    }

    let bytes_per_i32 = size_of::<i32>();

    let total_input_bytes = flat_data.len() * bytes_per_i32;

    println!("Total integer symbols: {}", flat_data.len());
    println!("Total raw bytes (i32): {}", total_input_bytes);

    // encode each stream with rANS and collect sizes
    let mut encoded: Vec<(&str, Vec<u8>, Vec<u8>, u32)> = Vec::new();
    let mut total_rans_bytes: usize = 0;

    let pb = ProgressBar::new(streams.len() as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}"
        )?
            .progress_chars("=>-"),
    );

    let mut i = 0;
    println!("Encoding streams:");
    for (name, original) in &streams {
        pb.set_message(format!("encoding attribute: {}", name));

        //per attribute-fine-tuned quantization step size and err NOTE! MUST MATCH WITH DECODER!
        let (step, err, mant_scale) = if i < 3 {
            //position (semi-resilient to quant)
            (1 << 11, i32::MAX / 4, 14u32)
        } else if  i > 51  {
            //opacity, scale, rotation (non-resilient to quant)
            (1 << 10, 512, 16u32)
        } else {
            //spherical harmonics (very resilient to quant)
            (1 << 19, 1 << 19, 13u32)
        };

        let (code, (raw_bytes, num_raw)) = encode_stream(&original, step as f32, err, mant_scale);


        let stream_size = code.len() + raw_bytes.len() ;
        total_rans_bytes += stream_size;

        encoded.push((name, code, raw_bytes, num_raw));
        i += 1;

        pb.inc(1);
    }
    pb.finish_with_message("Encoding complete");


    let pb = ProgressBar::new(streams.len() as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}"
        )?
            .progress_chars("=>-"),
    );
    //write encoded data to file
    println!("Writing .gsz file:");
    let file = File::create(file_path.clone())?;
    let mut writer = std::io::BufWriter::new(file);

    let scene_len = num_gaussians as u32;
    writer.write_all(&scene_len.to_le_bytes())?;

    for (name, code, raw_bytes, num_raw) in encoded.iter() {
        pb.set_message(format!("writing attribute: {}", name));

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

        //TODO mirror in decoder
        let raw_len = *num_raw;
        assert!(raw_len <= u32::MAX, "raw bytes length too large");
        writer.write_all(&raw_len.to_le_bytes())?;

        pb.inc(1);
    }
    pb.finish_with_message("Writing complete");

    writer.flush()?;

    println!(
        "\nTotal rANS coded size (all streams): {} bytes",
        total_rans_bytes
    );

    let input_f = total_input_bytes as f32;
    let rans_f = total_rans_bytes as f32;
    // let deflate_f = deflate_bytes as f32;

    // let input_f = ((s_xyz_x.len() + s_xyz_y.len() + s_xyz_z.len()) * size_of::<f32>()) as f32;
    // let rans_f = (encoded[0].1.len() + encoded[0].2.len() +
    //     encoded[1].1.len() + encoded[1].2.len() +
    //     encoded[2].1.len() + encoded[2].2.len())
    //     as f32;
    // //
    let rans_vs_raw = (input_f - rans_f) / input_f * 100.0;
    // let rans_vs_deflate = (deflate_f - rans_f) / deflate_f * 100.0;

    println!(
        "Percent improvement rANS vs raw:      {:.2} %",
        rans_vs_raw
    );
    // // deflate baseline on flattened data
    // // let deflate_code = compress_i32_vec(flat_data.clone())?;
    // // let deflate_bytes = deflate_code.len();
    // // println!("DEFLATE (LZ77 + Huffman) code size: {} bytes\n", deflate_bytes);
    // // println!(
    // //     "Percent improvement rANS vs DEFLATE: {:.2} %",
    // //     rans_vs_deflate
    // // );


    Ok(())
}
