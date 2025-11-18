use gaussian_parser::load_gaussians_from_ply;
use gaussian_sorter::generate_morton_code;
use rans_coding::{RansEnc, RansDec};
use deflate_coder::compress_i32_vec; // decompress not needed here

fn encode_stream(data: &Vec<i32>) -> (Vec<u8>, Vec<u8>) {
    // rough capacity based on input size
    let bytes_per_i32 = std::mem::size_of::<i32>();
    let est_bytes = data.len() * bytes_per_i32;
    let mut encoder = RansEnc::new(est_bytes * 2);

    // rANS encode single integer stream
    encoder.encode_values(data)
}

fn delta_encode(v: &mut [i32]) {
    if v.len() < 2 { return; }
    for i in (1..v.len()).rev() {
        v[i] = v[i] - v[i-1];
    }
}

fn delta_decode(v: &mut [i32]) {
    if v.len() < 2 { return; }
    for i in 1..v.len() {
        v[i] = v[i] + v[i-1];
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // load ply path from args
    let path: String = std::env::args().nth(1).expect("Missing .ply file path");
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
    let mut s_xyz_x: Vec<i32> = Vec::with_capacity(num_gaussians);
    let mut s_xyz_y: Vec<i32> = Vec::with_capacity(num_gaussians);
    let mut s_xyz_z: Vec<i32> = Vec::with_capacity(num_gaussians);

    // normals streams (optional, presence + components)
    // let mut s_normals_present: Vec<i32> = Vec::with_capacity(num_gaussians);
    // let mut s_normals_x: Vec<i32> = Vec::with_capacity(num_gaussians);
    // let mut s_normals_y: Vec<i32> = Vec::with_capacity(num_gaussians);
    // let mut s_normals_z: Vec<i32> = Vec::with_capacity(num_gaussians);

    // spherical harmonics streams
    let mut s_sh_dc_r: Vec<i32> = Vec::with_capacity(num_gaussians);
    let mut s_sh_dc_g: Vec<i32> = Vec::with_capacity(num_gaussians);
    let mut s_sh_dc_b: Vec<i32> = Vec::with_capacity(num_gaussians);

    // opacity stream
    let mut s_opacity: Vec<i32> = Vec::with_capacity(num_gaussians);

    // scale streams
    let mut s_scale_x: Vec<i32> = Vec::with_capacity(num_gaussians);
    let mut s_scale_y: Vec<i32> = Vec::with_capacity(num_gaussians);
    let mut s_scale_z: Vec<i32> = Vec::with_capacity(num_gaussians);

    // rotation quaternion streams
    let mut s_rot_x: Vec<i32> = Vec::with_capacity(num_gaussians);
    let mut s_rot_y: Vec<i32> = Vec::with_capacity(num_gaussians);
    let mut s_rot_z: Vec<i32> = Vec::with_capacity(num_gaussians);
    let mut s_rot_w: Vec<i32> = Vec::with_capacity(num_gaussians);

    // convert all gaussian fields to i32 streams (float cast)
    for g in &scene.gaussians {
        // xyz -> three streams
        s_xyz_x.push(g.xyz[0] as i32);
        s_xyz_y.push(g.xyz[1] as i32);
        s_xyz_z.push(g.xyz[2] as i32);

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
        s_sh_dc_r.push(g.sh_dc[0] as i32);
        s_sh_dc_g.push(g.sh_dc[1] as i32);
        s_sh_dc_b.push(g.sh_dc[2] as i32);

        // opacity -> single stream
        s_opacity.push(g.opacity as i32);

        // scale -> 3 streams (log-scales as-is)
        s_scale_x.push(g.scale[0] as i32);
        s_scale_y.push(g.scale[1] as i32);
        s_scale_z.push(g.scale[2] as i32);

        // rotation quaternion -> 4 streams
        s_rot_x.push(g.rot[0] as i32);
        s_rot_y.push(g.rot[1] as i32);
        s_rot_z.push(g.rot[2] as i32);
        s_rot_w.push(g.rot[3] as i32);
    }

    // vector to store streams
    let mut streams: Vec<(&'static str, Vec<i32>)> = Vec::new();

    // add position streams
    streams.push(("xyz_x", s_xyz_x));
    streams.push(("xyz_y", s_xyz_y));
    streams.push(("xyz_z", s_xyz_z));

    // add spherical harmonics streams
    streams.push(("sh_dc_r", s_sh_dc_r));
    streams.push(("sh_dc_g", s_sh_dc_g));
    streams.push(("sh_dc_b", s_sh_dc_b));

    //
    // for (j, v) in s_sh_rest.into_iter().enumerate() {
    //     let name = Box::leak(format!("sh_rest_{}", j).into_boxed_str());
    //     streams.push((name, v));
    // }

    // add opacity stream
    streams.push(("opacity", s_opacity));

    // add scale streams
    streams.push(("scale_x", s_scale_x));
    streams.push(("scale_y", s_scale_y));
    streams.push(("scale_z", s_scale_z));

    // add rotation streams
    streams.push(("rot_x", s_rot_x));
    streams.push(("rot_y", s_rot_y));
    streams.push(("rot_z", s_rot_z));
    streams.push(("rot_w", s_rot_w));

    println!("Number of integer streams: {}", streams.len());

    // flatten all integer data for a deflate baseline
    let mut flat_data: Vec<i32> = Vec::new();
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

    for (name, original) in &streams {
        let mut data = original.clone();
        delta_encode(&mut data);
        let (mut code, raw_bytes) = encode_stream(&data);

        let mn = data.iter().min().unwrap();
        let mx = data.iter().max().unwrap();
        println!("{} delta range = [{}, {}]", name, mn, mx);

        let stream_size = code.len() + raw_bytes.len();
        total_rans_bytes += stream_size;

        encoded.push((name, code, raw_bytes, data.len()));
    }

    println!(
        "\nTotal rANS coded size (all streams): {} bytes",
        total_rans_bytes
    );

    // deflate baseline on flattened data
    // let deflate_code = compress_i32_vec(flat_data.clone())?;
    // let deflate_bytes = deflate_code.len();
    // println!("DEFLATE (LZ77 + Huffman) code size: {} bytes\n", deflate_bytes);
    //
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

    // decode each stream and verify
    for (idx, (name, mut code, raw_bytes, len)) in encoded.into_iter().enumerate() {
        println!("{}", name);
        let mut decoder = RansDec::new(code.as_mut_slice(), raw_bytes);
        let mut decoded = decoder.decode_values(len);

        delta_decode(&mut decoded);

        let mut original = &streams[idx].1;
        assert_eq!(decoded, *original);
        if decoded != *original {
            println!("STREAM MISMATCH: {}", name);
        }
    }

    Ok(())
}
