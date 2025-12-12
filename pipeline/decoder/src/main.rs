use std::io;
use std::io::{ErrorKind, Read, Write};
use rans_coding::{RansDec};
use indicatif::{ProgressBar, ProgressStyle};


const SH_REST_LEN: usize = 45;

pub struct EncodedAttribute{
    pub coded: Vec<u8>,
    pub raw: Vec<u8>,
    pub raw_len: u32,
}


pub fn write_gaussians_to_ply(path: &str, gaussian: &Vec<Vec<f32>>, num_gaus: u32) -> Result<(), Box<dyn std::error::Error>> {
    let file = std::fs::File::create(path)?;
    let mut w = std::io::BufWriter::new(file);

    // --------- write PLY header (binary_little_endian) ----------
    writeln!(w, "ply")?;
    writeln!(w, "format binary_little_endian 1.0")?;
    writeln!(w, "element vertex {}", num_gaus)?;
    writeln!(w, "property float x")?;
    writeln!(w, "property float y")?;
    writeln!(w, "property float z")?;
    writeln!(w, "property float f_dc_0")?;
    writeln!(w, "property float f_dc_1")?;
    writeln!(w, "property float f_dc_2")?;

    for i in 0..SH_REST_LEN {
        writeln!(w, "property float f_rest_{}", i)?;
    }

    writeln!(w, "property float opacity")?;
    writeln!(w, "property float scale_0")?;
    writeln!(w, "property float scale_1")?;
    writeln!(w, "property float scale_2")?;
    writeln!(w, "property float rot_0")?;
    writeln!(w, "property float rot_1")?;
    writeln!(w, "property float rot_2")?;
    writeln!(w, "property float rot_3")?;
    writeln!(w, "end_header")?;

    let pb = ProgressBar::new(num_gaus as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}"
        )?
            .progress_chars("=>-"),
    );

    println!("Writing .ply file:");
    for i in 0..num_gaus as usize {
        let mut attr_idx = 0;
        // write x
        w.write_all(&gaussian[attr_idx][i].to_le_bytes())?;
        attr_idx += 1;
        //write y
        w.write_all(&gaussian[attr_idx][i].to_le_bytes())?;
        attr_idx += 1;
        //write z
        w.write_all(&gaussian[attr_idx][i].to_le_bytes())?;
        attr_idx += 1;

        //write sh_dc_0
        w.write_all(&gaussian[attr_idx][i].to_le_bytes())?;
        attr_idx += 1;
        //write sh_dc_1
        w.write_all(&gaussian[attr_idx][i].to_le_bytes())?;
        attr_idx += 1;
        //write sh_dc_2
        w.write_all(&gaussian[attr_idx][i].to_le_bytes())?;
        attr_idx += 1;

        for _ in 0..SH_REST_LEN {
            w.write_all(&gaussian[attr_idx][i].to_le_bytes())?;
            attr_idx += 1;
        }

        //write opacity
        w.write_all(&gaussian[attr_idx][i].to_le_bytes())?;
        attr_idx += 1;

        //scale
        w.write_all(&gaussian[attr_idx][i].to_le_bytes())?;
        attr_idx += 1;
        w.write_all(&gaussian[attr_idx][i].to_le_bytes())?;
        attr_idx += 1;
        w.write_all(&gaussian[attr_idx][i].to_le_bytes())?;
        attr_idx += 1;

        //rotation quaternion
        w.write_all(&gaussian[attr_idx][i].to_le_bytes())?;
        attr_idx += 1;
        w.write_all(&gaussian[attr_idx][i].to_le_bytes())?;
        attr_idx += 1;
        w.write_all(&gaussian[attr_idx][i].to_le_bytes())?;
        attr_idx += 1;
        w.write_all(&gaussian[attr_idx][i].to_le_bytes())?;
        pb.inc(1);
    }
    pb.finish_with_message("Finished Writing .ply");

    w.flush()?;
    Ok(())
}

pub fn read_gaussian_from_gsz(path: &str, coded_data: &mut Vec<EncodedAttribute>, scene_len: &mut u32) -> Result<(), Box<dyn std::error::Error>> {
    let file = std::fs::File::open(path)?;
    let mut reader = std::io::BufReader::new(file);

    let mut len_buf = [0u8; 4];
    reader.read_exact(&mut len_buf)?;
    *scene_len = u32::from_le_bytes(len_buf);

    let pb = ProgressBar::new(59u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}"
        )?
            .progress_chars("=>-"),
    );

    let mut i = 0;
    println!("Reading .gsz file:");
    loop {
        //read code length
        pb.set_message(format!("Reading attribute {}", i));

        match reader.read_exact(&mut len_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }
        let code_len = u32::from_le_bytes(len_buf);

        //read coded data
        let mut code_buf = vec![0u8; code_len as usize];
        reader.read_exact(&mut code_buf)?;

        //read raw data length
        reader.read_exact(&mut len_buf)?;
        let raw_len = u32::from_le_bytes(len_buf);

        //read raw data
        let mut raw_buf = vec![0u8; raw_len as usize];
        reader.read_exact(&mut raw_buf)?;

        //read raw len
        reader.read_exact(&mut len_buf)?;
        let raw_len = u32::from_le_bytes(len_buf);

        coded_data.push(EncodedAttribute {coded: code_buf, raw: raw_buf, raw_len});
        i += 1;
        pb.inc(1);
    }
    pb.finish_with_message("Finished Reading Attributes");


    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>>{
    let path: String = std::env::args().nth(1).expect("Missing .gsz file path");
    let ply_path: String = std::env::args().nth(2).expect("Missing .ply file path");
    println!("Input GSZ: {}", path);

   let mut coded_data : Vec<EncodedAttribute> = Vec::new();
   let mut scene_len : u32 = 0;

    read_gaussian_from_gsz(&path, &mut coded_data, &mut scene_len)?;


    let pb = ProgressBar::new(coded_data.len() as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}"
        )?
            .progress_chars("=>-"),
    );

    println!("Decoding .gsz file:");
    let mut res : Vec<Vec<f32>> = Vec::new();
    for (i, encoded_attribute) in coded_data.iter_mut().enumerate() {
        pb.set_message(format!("Decoding attribute {}", i));
        //attribute-specific fine tuned qauntization parameters NOTE! MUST MATCH WITH ENCODER!
        let (step, mant_scale) = if i < 3 {
            //position attribte (step size, scale bit)
            (1 << 11, 14u32)
        } else if  i > 51  {
            //opacity, rotation, scale attributes (step size, scale bit)
            (1 << 10, 16u32)
        } else {
            //spherical harmonics attributes (step size, scale bit)
            (1 << 19, 13u32)
        };
        let mut decoder = RansDec::new(&mut encoded_attribute.coded, &mut encoded_attribute.raw, encoded_attribute.raw_len, step as f32, mant_scale);
        let attr = decoder.decode_values(scene_len as usize);
        res.push(attr);
        pb.inc(1);
    }
    pb.finish_with_message("Finished Decoding Attributes");

    write_gaussians_to_ply(&ply_path, &res, scene_len)?;


    Ok(())
}
