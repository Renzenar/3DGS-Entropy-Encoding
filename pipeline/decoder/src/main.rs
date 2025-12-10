use std::io;
use std::io::{ErrorKind, Read, Write};
use rans_coding::{RansDec};

const SH_REST_LEN: usize = 45;

pub struct EncodedAttribute{
    pub coded: Vec<u8>,
    pub raw: Vec<u8>,
}


pub fn write_gaussians_to_ply(path: &str, gaussian: &Vec<Vec<f32>>, num_gaus: u32) -> io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    let mut w = std::io::BufWriter::new(file);


    // println!("Writing {} gaussians to ply file | gaussians in array {} ", num_gaus, g.len());

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
    }

    w.flush()?;
    Ok(())
}

pub fn read_gaussian_from_gsz(path: &str, coded_data: &mut Vec<EncodedAttribute>, scene_len: &mut u32) -> io::Result<()> {
    let file = std::fs::File::open(path)?;
    let mut reader = std::io::BufReader::new(file);
    // let mut coded_data : Vec<EncodedAttribute> = Vec::new();

    let mut len_buf = [0u8; 4];
    reader.read_exact(&mut len_buf)?;
    *scene_len = u32::from_le_bytes(len_buf);

    loop {
        //read code length

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

        coded_data.push(lib1::EncodedAttribute {coded: code_buf, raw: raw_buf});
    }


    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>>{
    let path: String = std::env::args().nth(1).expect("Missing .gsz file path");
    let ply_path: String = std::env::args().nth(2).expect("Missing .ply file path");
    println!("Input GSZ: {}", path);

   let mut coded_data : Vec<EncodedAttribute> = Vec::new();
   let mut scene_len : u32 = 0;

    read_gaussian_from_gsz(&path, &mut coded_data, &mut scene_len)?;

    println!("Coded data len {}, num gaussians {} ", coded_data.len(), scene_len );


    let mut res : Vec<Vec<f32>> = Vec::new();
    for mut encoded_attribute in coded_data{
        let mut decoder = RansDec::new(encoded_attribute.coded.as_mut_slice(), encoded_attribute.raw);
        let attr = decoder.decode_values(scene_len as usize);
        res.push(attr);
    }

    write_gaussians_to_ply(&ply_path, &res, scene_len)?;


    Ok(())
}
