use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use gaussian_parser::Scene;
use gaussian_sorter::generate_morton_code;
use gaussian_types::Gaussian;
use serde::{Deserialize, Serialize};

const META_FILE: &str = "meta.json";
const OCCUPANCY_FILE: &str = "occupancy.bin";
const GEOM_X_FILE: &str = "geom_x.bin";
const GEOM_Y_FILE: &str = "geom_y.bin";
const GEOM_Z_FILE: &str = "geom_z.bin";
const SH_DC_R_FILE: &str = "sh_dc_r.bin";
const SH_DC_G_FILE: &str = "sh_dc_g.bin";
const SH_DC_B_FILE: &str = "sh_dc_b.bin";

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QuantParams {
    pub min: f32,
    pub max: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PackedRasterScene {
    pub num_gaussians: u32,
    pub width: u32,
    pub height: u32,
    pub sh_rest_len: u32,
    pub occupancy: Vec<u8>,
    pub geom_x: Vec<u16>,
    pub geom_y: Vec<u16>,
    pub geom_z: Vec<u16>,
    pub sh_dc_r: Vec<u16>,
    pub sh_dc_g: Vec<u16>,
    pub sh_dc_b: Vec<u16>,
    pub geom_quant: [QuantParams; 3],
    pub sh_dc_quant: [QuantParams; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct PackedRasterMetadata {
    num_gaussians: u32,
    width: u32,
    height: u32,
    sh_rest_len: u32,
    geom_quant: [QuantParams; 3],
    sh_dc_quant: [QuantParams; 3],
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChannelDeltaStats {
    pub count: usize,
    pub mean_abs: f64,
    pub std_abs: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NeighborDeltaStats {
    pub xyz: [ChannelDeltaStats; 3],
    pub sh_dc: [ChannelDeltaStats; 3],
}

#[derive(Debug, Clone, PartialEq)]
pub struct CorrelationStats {
    pub count: usize,
    pub pearson_r: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlaneStats {
    pub name: &'static str,
    pub min: u32,
    pub max: u32,
    pub entropy_bits_per_symbol: f64,
    pub neighbor_delta: ChannelDeltaStats,
    pub horizontal_correlation: CorrelationStats,
    pub vertical_correlation: CorrelationStats,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ByteSplitPlane {
    pub name: &'static str,
    pub hi: Vec<u8>,
    pub lo: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ByteSplitPlaneStats {
    pub name: &'static str,
    pub base_u16: PlaneStats,
    pub hi: PlaneStats,
    pub lo: PlaneStats,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GeomHiPngPaths {
    pub geom_x_hi: PathBuf,
    pub geom_y_hi: PathBuf,
    pub geom_z_hi: PathBuf,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GeomHiRgbPngPath {
    pub geom_hi_rgb: PathBuf,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ShDcHiRgbPngPath {
    pub sh_dc_hi_rgb: PathBuf,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GeomHiPngExperimentResult {
    pub image_sizes: [u64; 3],
    pub total_coded_bytes: u64,
    pub xyz_rmse: [f64; 3],
}

#[derive(Debug, Clone, PartialEq)]
pub struct RoundtripDiagnostics {
    pub xyz_rmse: [f64; 3],
    pub sh_dc_rmse: [f64; 3],
    pub original_neighbor_deltas: NeighborDeltaStats,
    pub reconstructed_neighbor_deltas: NeighborDeltaStats,
    pub plane_stats: Vec<PlaneStats>,
    pub byte_split_plane_stats: Vec<ByteSplitPlaneStats>,
}

pub fn pack_scene_to_raster(scene: &Scene, width: u32) -> PackedRasterScene {
    assert!(width > 0, "atlas width must be greater than zero");

    let mut sorted = scene.gaussians.clone();
    sorted.sort_unstable_by_key(|g| generate_morton_code(g, &scene.mins, &scene.maxes));

    let num_gaussians = sorted.len() as u32;
    let sh_rest_len = sorted.first().map(|g| g.sh_rest.len() as u32).unwrap_or(0);
    let height = if num_gaussians == 0 {
        0
    } else {
        num_gaussians.div_ceil(width)
    };
    let plane_len = (width as usize) * (height as usize);

    let (geom_quant, sh_dc_quant) = compute_quant_params(&sorted);
    let mut packed = PackedRasterScene {
        num_gaussians,
        width,
        height,
        sh_rest_len,
        occupancy: vec![0; plane_len],
        geom_x: vec![0; plane_len],
        geom_y: vec![0; plane_len],
        geom_z: vec![0; plane_len],
        sh_dc_r: vec![0; plane_len],
        sh_dc_g: vec![0; plane_len],
        sh_dc_b: vec![0; plane_len],
        geom_quant,
        sh_dc_quant,
    };

    for (idx, gaussian) in sorted.iter().enumerate() {
        packed.occupancy[idx] = 1;
        packed.geom_x[idx] = quantize_to_u16(gaussian.xyz[0], packed.geom_quant[0]);
        packed.geom_y[idx] = quantize_to_u16(gaussian.xyz[1], packed.geom_quant[1]);
        packed.geom_z[idx] = quantize_to_u16(gaussian.xyz[2], packed.geom_quant[2]);
        packed.sh_dc_r[idx] = quantize_to_u16(gaussian.sh_dc[0], packed.sh_dc_quant[0]);
        packed.sh_dc_g[idx] = quantize_to_u16(gaussian.sh_dc[1], packed.sh_dc_quant[1]);
        packed.sh_dc_b[idx] = quantize_to_u16(gaussian.sh_dc[2], packed.sh_dc_quant[2]);
    }

    packed
}

pub fn unpack_raster_to_scene(packed: &PackedRasterScene) -> Scene {
    let mut gaussians = Vec::with_capacity(packed.num_gaussians as usize);

    for idx in 0..packed.occupancy.len() {
        if packed.occupancy[idx] == 0 {
            continue;
        }

        let xyz = [
            dequantize_from_u16(packed.geom_x[idx], packed.geom_quant[0]),
            dequantize_from_u16(packed.geom_y[idx], packed.geom_quant[1]),
            dequantize_from_u16(packed.geom_z[idx], packed.geom_quant[2]),
        ];
        let sh_dc = [
            dequantize_from_u16(packed.sh_dc_r[idx], packed.sh_dc_quant[0]),
            dequantize_from_u16(packed.sh_dc_g[idx], packed.sh_dc_quant[1]),
            dequantize_from_u16(packed.sh_dc_b[idx], packed.sh_dc_quant[2]),
        ];

        gaussians.push(Gaussian {
            xyz,
            normals: None,
            sh_dc,
            sh_rest: vec![0.0; packed.sh_rest_len as usize],
            opacity: 1.0,
            scale: [0.0; 3],
            rot: [0.0, 0.0, 0.0, 1.0],
        });
    }

    gaussians.truncate(packed.num_gaussians as usize);
    let (mins, maxes) = compute_xyz_bounds(&gaussians);
    let mut meta = HashMap::new();
    meta.insert("format".to_string(), "packed_raster_v1".to_string());
    meta.insert("vertex_count".to_string(), gaussians.len().to_string());

    Scene {
        gaussians,
        meta,
        mins,
        maxes,
    }
}

pub fn write_packed_raster_scene_dir(
    packed: &PackedRasterScene,
    output_dir: impl AsRef<Path>,
) -> std::io::Result<()> {
    let output_dir = output_dir.as_ref();
    fs::create_dir_all(output_dir)?;

    let metadata = PackedRasterMetadata {
        num_gaussians: packed.num_gaussians,
        width: packed.width,
        height: packed.height,
        sh_rest_len: packed.sh_rest_len,
        geom_quant: packed.geom_quant,
        sh_dc_quant: packed.sh_dc_quant,
    };

    let meta_path = output_dir.join(META_FILE);
    let meta_writer = BufWriter::new(File::create(meta_path)?);
    serde_json::to_writer_pretty(meta_writer, &metadata)?;

    write_u8_plane(output_dir.join(OCCUPANCY_FILE), &packed.occupancy)?;
    write_u16_plane(output_dir.join(GEOM_X_FILE), &packed.geom_x)?;
    write_u16_plane(output_dir.join(GEOM_Y_FILE), &packed.geom_y)?;
    write_u16_plane(output_dir.join(GEOM_Z_FILE), &packed.geom_z)?;
    write_u16_plane(output_dir.join(SH_DC_R_FILE), &packed.sh_dc_r)?;
    write_u16_plane(output_dir.join(SH_DC_G_FILE), &packed.sh_dc_g)?;
    write_u16_plane(output_dir.join(SH_DC_B_FILE), &packed.sh_dc_b)?;

    Ok(())
}

pub fn read_packed_raster_scene_dir(
    input_dir: impl AsRef<Path>,
) -> Result<PackedRasterScene, Box<dyn std::error::Error>> {
    let input_dir = input_dir.as_ref();
    let metadata: PackedRasterMetadata = serde_json::from_reader(BufReader::new(File::open(
        input_dir.join(META_FILE),
    )?))?;

    let plane_len = (metadata.width as usize) * (metadata.height as usize);

    let occupancy = read_u8_plane(input_dir.join(OCCUPANCY_FILE), plane_len)?;
    let geom_x = read_u16_plane(input_dir.join(GEOM_X_FILE), plane_len)?;
    let geom_y = read_u16_plane(input_dir.join(GEOM_Y_FILE), plane_len)?;
    let geom_z = read_u16_plane(input_dir.join(GEOM_Z_FILE), plane_len)?;
    let sh_dc_r = read_u16_plane(input_dir.join(SH_DC_R_FILE), plane_len)?;
    let sh_dc_g = read_u16_plane(input_dir.join(SH_DC_G_FILE), plane_len)?;
    let sh_dc_b = read_u16_plane(input_dir.join(SH_DC_B_FILE), plane_len)?;

    Ok(PackedRasterScene {
        num_gaussians: metadata.num_gaussians,
        width: metadata.width,
        height: metadata.height,
        sh_rest_len: metadata.sh_rest_len,
        occupancy,
        geom_x,
        geom_y,
        geom_z,
        sh_dc_r,
        sh_dc_g,
        sh_dc_b,
        geom_quant: metadata.geom_quant,
        sh_dc_quant: metadata.sh_dc_quant,
    })
}

pub fn write_scene_to_ply(
    path: impl AsRef<Path>,
    scene: &Scene,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = path.as_ref();
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    let num_gaussians = scene.gaussians.len();
    let sh_rest_len = scene.gaussians.first().map(|g| g.sh_rest.len()).unwrap_or(0);

    writeln!(writer, "ply")?;
    writeln!(writer, "format binary_little_endian 1.0")?;
    writeln!(writer, "element vertex {}", num_gaussians)?;
    writeln!(writer, "property float x")?;
    writeln!(writer, "property float y")?;
    writeln!(writer, "property float z")?;
    writeln!(writer, "property float f_dc_0")?;
    writeln!(writer, "property float f_dc_1")?;
    writeln!(writer, "property float f_dc_2")?;
    for i in 0..sh_rest_len {
        writeln!(writer, "property float f_rest_{}", i)?;
    }
    writeln!(writer, "property float opacity")?;
    writeln!(writer, "property float scale_0")?;
    writeln!(writer, "property float scale_1")?;
    writeln!(writer, "property float scale_2")?;
    writeln!(writer, "property float rot_0")?;
    writeln!(writer, "property float rot_1")?;
    writeln!(writer, "property float rot_2")?;
    writeln!(writer, "property float rot_3")?;
    writeln!(writer, "end_header")?;

    for gaussian in &scene.gaussians {
        writer.write_all(&gaussian.xyz[0].to_le_bytes())?;
        writer.write_all(&gaussian.xyz[1].to_le_bytes())?;
        writer.write_all(&gaussian.xyz[2].to_le_bytes())?;
        writer.write_all(&gaussian.sh_dc[0].to_le_bytes())?;
        writer.write_all(&gaussian.sh_dc[1].to_le_bytes())?;
        writer.write_all(&gaussian.sh_dc[2].to_le_bytes())?;
        for value in &gaussian.sh_rest {
            writer.write_all(&value.to_le_bytes())?;
        }
        writer.write_all(&gaussian.opacity.to_le_bytes())?;
        writer.write_all(&gaussian.scale[0].to_le_bytes())?;
        writer.write_all(&gaussian.scale[1].to_le_bytes())?;
        writer.write_all(&gaussian.scale[2].to_le_bytes())?;
        writer.write_all(&gaussian.rot[0].to_le_bytes())?;
        writer.write_all(&gaussian.rot[1].to_le_bytes())?;
        writer.write_all(&gaussian.rot[2].to_le_bytes())?;
        writer.write_all(&gaussian.rot[3].to_le_bytes())?;
    }

    writer.flush()?;
    Ok(())
}

pub fn packed_raster_raw_payload_bytes(packed: &PackedRasterScene) -> usize {
    packed.occupancy.len()
        + (packed.geom_x.len() * std::mem::size_of::<u16>())
        + (packed.geom_y.len() * std::mem::size_of::<u16>())
        + (packed.geom_z.len() * std::mem::size_of::<u16>())
        + (packed.sh_dc_r.len() * std::mem::size_of::<u16>())
        + (packed.sh_dc_g.len() * std::mem::size_of::<u16>())
        + (packed.sh_dc_b.len() * std::mem::size_of::<u16>())
}

pub fn write_debug_preview_pngs(
    packed: &PackedRasterScene,
    output_dir: impl AsRef<Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let output_dir = output_dir.as_ref();
    fs::create_dir_all(output_dir)?;

    write_preview_u8_png(
        output_dir.join("occupancy.png"),
        packed.width,
        packed.height,
        &packed.occupancy,
        &packed.occupancy,
    )?;
    write_preview_u16_png(
        output_dir.join("geom_x.png"),
        packed.width,
        packed.height,
        &packed.geom_x,
        &packed.occupancy,
    )?;
    write_preview_u16_png(
        output_dir.join("geom_y.png"),
        packed.width,
        packed.height,
        &packed.geom_y,
        &packed.occupancy,
    )?;
    write_preview_u16_png(
        output_dir.join("geom_z.png"),
        packed.width,
        packed.height,
        &packed.geom_z,
        &packed.occupancy,
    )?;
    write_preview_u16_png(
        output_dir.join("sh_dc_r.png"),
        packed.width,
        packed.height,
        &packed.sh_dc_r,
        &packed.occupancy,
    )?;
    write_preview_u16_png(
        output_dir.join("sh_dc_g.png"),
        packed.width,
        packed.height,
        &packed.sh_dc_g,
        &packed.occupancy,
    )?;
    write_preview_u16_png(
        output_dir.join("sh_dc_b.png"),
        packed.width,
        packed.height,
        &packed.sh_dc_b,
        &packed.occupancy,
    )?;

    let byte_split = split_packed_u16_planes(packed);
    for plane in &byte_split {
        write_preview_u8_png(
            output_dir.join(format!("{}_hi.png", plane.name)),
            packed.width,
            packed.height,
            &plane.hi,
            &packed.occupancy,
        )?;
        write_preview_u8_png(
            output_dir.join(format!("{}_lo.png", plane.name)),
            packed.width,
            packed.height,
            &plane.lo,
            &packed.occupancy,
        )?;
    }

    Ok(())
}

pub fn write_byte_split_planes_dir(
    packed: &PackedRasterScene,
    output_dir: impl AsRef<Path>,
) -> std::io::Result<()> {
    let output_dir = output_dir.as_ref();
    fs::create_dir_all(output_dir)?;
    let byte_split = split_packed_u16_planes(packed);
    for plane in &byte_split {
        write_u8_plane(output_dir.join(format!("{}_hi.bin", plane.name)), &plane.hi)?;
        write_u8_plane(output_dir.join(format!("{}_lo.bin", plane.name)), &plane.lo)?;
    }
    Ok(())
}

pub fn write_geom_hi_pngs(
    packed: &PackedRasterScene,
    output_dir: impl AsRef<Path>,
) -> Result<GeomHiPngPaths, Box<dyn std::error::Error>> {
    let output_dir = output_dir.as_ref();
    fs::create_dir_all(output_dir)?;
    let geom_x_hi = output_dir.join("geom_x_hi.png");
    let geom_y_hi = output_dir.join("geom_y_hi.png");
    let geom_z_hi = output_dir.join("geom_z_hi.png");

    let (geom_x_hi_bytes, _) = split_u16_plane_to_bytes(&packed.geom_x);
    let (geom_y_hi_bytes, _) = split_u16_plane_to_bytes(&packed.geom_y);
    let (geom_z_hi_bytes, _) = split_u16_plane_to_bytes(&packed.geom_z);

    write_lossless_u8_png(&geom_x_hi, packed.width, packed.height, &geom_x_hi_bytes)?;
    write_lossless_u8_png(&geom_y_hi, packed.width, packed.height, &geom_y_hi_bytes)?;
    write_lossless_u8_png(&geom_z_hi, packed.width, packed.height, &geom_z_hi_bytes)?;

    Ok(GeomHiPngPaths {
        geom_x_hi,
        geom_y_hi,
        geom_z_hi,
    })
}

pub fn read_u8_png(
    path: impl AsRef<Path>,
    expected_width: u32,
    expected_height: u32,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let file = File::open(path.as_ref())?;
    let decoder = png::Decoder::new(BufReader::new(file));
    let mut reader = decoder.read_info()?;
    let info = reader.info();
    if info.width != expected_width || info.height != expected_height {
        return Err(format!(
            "png dimensions mismatch: expected {}x{}, got {}x{}",
            expected_width, expected_height, info.width, info.height
        )
        .into());
    }
    if info.color_type != png::ColorType::Grayscale || info.bit_depth != png::BitDepth::Eight {
        return Err("expected 8-bit grayscale png".into());
    }

    let mut buf = vec![0u8; reader.output_buffer_size()];
    let frame = reader.next_frame(&mut buf)?;
    Ok(buf[..frame.buffer_size()].to_vec())
}

pub fn write_geom_hi_rgb_png(
    packed: &PackedRasterScene,
    output_dir: impl AsRef<Path>,
) -> Result<GeomHiRgbPngPath, Box<dyn std::error::Error>> {
    let output_dir = output_dir.as_ref();
    fs::create_dir_all(output_dir)?;
    let geom_hi_rgb = output_dir.join("geom_hi_rgb.png");

    let (geom_x_hi_bytes, _) = split_u16_plane_to_bytes(&packed.geom_x);
    let (geom_y_hi_bytes, _) = split_u16_plane_to_bytes(&packed.geom_y);
    let (geom_z_hi_bytes, _) = split_u16_plane_to_bytes(&packed.geom_z);

    let mut rgb = Vec::with_capacity(geom_x_hi_bytes.len() * 3);
    for i in 0..geom_x_hi_bytes.len() {
        rgb.push(geom_x_hi_bytes[i]);
        rgb.push(geom_y_hi_bytes[i]);
        rgb.push(geom_z_hi_bytes[i]);
    }

    write_rgb_png(&geom_hi_rgb, packed.width, packed.height, &rgb)?;
    Ok(GeomHiRgbPngPath { geom_hi_rgb })
}

pub fn write_sh_dc_hi_rgb_png(
    packed: &PackedRasterScene,
    output_dir: impl AsRef<Path>,
) -> Result<ShDcHiRgbPngPath, Box<dyn std::error::Error>> {
    let output_dir = output_dir.as_ref();
    fs::create_dir_all(output_dir)?;
    let sh_dc_hi_rgb = output_dir.join("sh_dc_hi_rgb.png");

    let (sh_dc_r_hi_bytes, _) = split_u16_plane_to_bytes(&packed.sh_dc_r);
    let (sh_dc_g_hi_bytes, _) = split_u16_plane_to_bytes(&packed.sh_dc_g);
    let (sh_dc_b_hi_bytes, _) = split_u16_plane_to_bytes(&packed.sh_dc_b);

    write_rgb_png_from_planes(
        &sh_dc_hi_rgb,
        packed.width,
        packed.height,
        &sh_dc_r_hi_bytes,
        &sh_dc_g_hi_bytes,
        &sh_dc_b_hi_bytes,
    )?;

    Ok(ShDcHiRgbPngPath { sh_dc_hi_rgb })
}

pub fn read_rgb_png(
    path: impl AsRef<Path>,
    expected_width: u32,
    expected_height: u32,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let file = File::open(path.as_ref())?;
    let decoder = png::Decoder::new(BufReader::new(file));
    let mut reader = decoder.read_info()?;
    let info = reader.info();
    if info.width != expected_width || info.height != expected_height {
        return Err(format!(
            "png dimensions mismatch: expected {}x{}, got {}x{}",
            expected_width, expected_height, info.width, info.height
        )
        .into());
    }
    if info.color_type != png::ColorType::Rgb || info.bit_depth != png::BitDepth::Eight {
        return Err("expected 8-bit rgb png".into());
    }

    let mut buf = vec![0u8; reader.output_buffer_size()];
    let frame = reader.next_frame(&mut buf)?;
    Ok(buf[..frame.buffer_size()].to_vec())
}

pub fn load_geom_hi_pngs(
    png_dir: impl AsRef<Path>,
    width: u32,
    height: u32,
) -> Result<[Vec<u8>; 3], Box<dyn std::error::Error>> {
    let png_dir = png_dir.as_ref();
    Ok([
        read_u8_png(png_dir.join("geom_x_hi.png"), width, height)?,
        read_u8_png(png_dir.join("geom_y_hi.png"), width, height)?,
        read_u8_png(png_dir.join("geom_z_hi.png"), width, height)?,
    ])
}

pub fn load_geom_hi_rgb_png(
    png_dir: impl AsRef<Path>,
    width: u32,
    height: u32,
) -> Result<[Vec<u8>; 3], Box<dyn std::error::Error>> {
    read_rgb_png_to_planes(png_dir.as_ref().join("geom_hi_rgb.png"), width, height)
}

pub fn read_rgb_png_to_planes(
    path: impl AsRef<Path>,
    width: u32,
    height: u32,
) -> Result<[Vec<u8>; 3], Box<dyn std::error::Error>> {
    let rgb = read_rgb_png(path, width, height)?;
    let pixel_count = (width as usize) * (height as usize);
    if rgb.len() != pixel_count * 3 {
        return Err(format!(
            "rgb png byte length mismatch: expected {}, got {}",
            pixel_count * 3,
            rgb.len()
        )
        .into());
    }

    let mut x = Vec::with_capacity(pixel_count);
    let mut y = Vec::with_capacity(pixel_count);
    let mut z = Vec::with_capacity(pixel_count);
    for chunk in rgb.chunks_exact(3) {
        x.push(chunk[0]);
        y.push(chunk[1]);
        z.push(chunk[2]);
    }
    Ok([x, y, z])
}

pub fn write_rgb_png_from_planes(
    path: impl AsRef<Path>,
    width: u32,
    height: u32,
    r: &[u8],
    g: &[u8],
    b: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(r.len(), g.len(), "rgb plane lengths must match");
    assert_eq!(r.len(), b.len(), "rgb plane lengths must match");

    let mut rgb = Vec::with_capacity(r.len() * 3);
    for i in 0..r.len() {
        rgb.push(r[i]);
        rgb.push(g[i]);
        rgb.push(b[i]);
    }

    write_rgb_png(path, width, height, &rgb)
}

pub fn recombine_bytes_to_u16(hi: &[u8], lo: &[u8]) -> Vec<u16> {
    assert_eq!(hi.len(), lo.len(), "hi/lo planes must have equal length");
    hi.iter()
        .zip(lo.iter())
        .map(|(&hi_byte, &lo_byte)| u16::from_le_bytes([lo_byte, hi_byte]))
        .collect()
}

pub fn rebuild_packed_scene_with_decoded_geom_hi(
    packed: &PackedRasterScene,
    decoded_geom_hi: [&[u8]; 3],
) -> PackedRasterScene {
    let (_, geom_x_lo) = split_u16_plane_to_bytes(&packed.geom_x);
    let (_, geom_y_lo) = split_u16_plane_to_bytes(&packed.geom_y);
    let (_, geom_z_lo) = split_u16_plane_to_bytes(&packed.geom_z);

    let mut rebuilt = packed.clone();
    rebuilt.geom_x = recombine_bytes_to_u16(decoded_geom_hi[0], &geom_x_lo);
    rebuilt.geom_y = recombine_bytes_to_u16(decoded_geom_hi[1], &geom_y_lo);
    rebuilt.geom_z = recombine_bytes_to_u16(decoded_geom_hi[2], &geom_z_lo);
    rebuilt
}

pub fn rebuild_packed_scene_with_decoded_sh_dc_hi(
    packed: &PackedRasterScene,
    decoded_sh_dc_hi: [&[u8]; 3],
) -> PackedRasterScene {
    let (_, sh_dc_r_lo) = split_u16_plane_to_bytes(&packed.sh_dc_r);
    let (_, sh_dc_g_lo) = split_u16_plane_to_bytes(&packed.sh_dc_g);
    let (_, sh_dc_b_lo) = split_u16_plane_to_bytes(&packed.sh_dc_b);

    let mut rebuilt = packed.clone();
    rebuilt.sh_dc_r = recombine_bytes_to_u16(decoded_sh_dc_hi[0], &sh_dc_r_lo);
    rebuilt.sh_dc_g = recombine_bytes_to_u16(decoded_sh_dc_hi[1], &sh_dc_g_lo);
    rebuilt.sh_dc_b = recombine_bytes_to_u16(decoded_sh_dc_hi[2], &sh_dc_b_lo);
    rebuilt
}

pub fn evaluate_geom_hi_png_roundtrip(
    original_sorted: &[Gaussian],
    packed: &PackedRasterScene,
    png_dir: impl AsRef<Path>,
) -> Result<GeomHiPngExperimentResult, Box<dyn std::error::Error>> {
    let png_dir = png_dir.as_ref();
    let decoded_hi = load_geom_hi_pngs(png_dir, packed.width, packed.height)?;
    let rebuilt = rebuild_packed_scene_with_decoded_geom_hi(
        packed,
        [&decoded_hi[0], &decoded_hi[1], &decoded_hi[2]],
    );
    let reconstructed = unpack_raster_to_scene(&rebuilt);

    let xyz_rmse = [
        rmse_for_channel(original_sorted, &reconstructed.gaussians, |g| g.xyz[0]),
        rmse_for_channel(original_sorted, &reconstructed.gaussians, |g| g.xyz[1]),
        rmse_for_channel(original_sorted, &reconstructed.gaussians, |g| g.xyz[2]),
    ];

    let x_size = fs::metadata(png_dir.join("geom_x_hi.png"))?.len();
    let y_size = fs::metadata(png_dir.join("geom_y_hi.png"))?.len();
    let z_size = fs::metadata(png_dir.join("geom_z_hi.png"))?.len();
    let image_sizes = [x_size, y_size, z_size];
    let total_coded_bytes = image_sizes.iter().sum();

    Ok(GeomHiPngExperimentResult {
        image_sizes,
        total_coded_bytes,
        xyz_rmse,
    })
}

pub fn evaluate_geom_hi_rgb_png_roundtrip(
    original_sorted: &[Gaussian],
    packed: &PackedRasterScene,
    png_dir: impl AsRef<Path>,
) -> Result<GeomHiPngExperimentResult, Box<dyn std::error::Error>> {
    let png_dir = png_dir.as_ref();
    let decoded_hi = load_geom_hi_rgb_png(png_dir, packed.width, packed.height)?;
    let rebuilt = rebuild_packed_scene_with_decoded_geom_hi(
        packed,
        [&decoded_hi[0], &decoded_hi[1], &decoded_hi[2]],
    );
    let reconstructed = unpack_raster_to_scene(&rebuilt);

    let xyz_rmse = [
        rmse_for_channel(original_sorted, &reconstructed.gaussians, |g| g.xyz[0]),
        rmse_for_channel(original_sorted, &reconstructed.gaussians, |g| g.xyz[1]),
        rmse_for_channel(original_sorted, &reconstructed.gaussians, |g| g.xyz[2]),
    ];

    let rgb_size = fs::metadata(png_dir.join("geom_hi_rgb.png"))?.len();
    Ok(GeomHiPngExperimentResult {
        image_sizes: [rgb_size, 0, 0],
        total_coded_bytes: rgb_size,
        xyz_rmse,
    })
}

pub fn compute_roundtrip_diagnostics(
    original_sorted: &[Gaussian],
    reconstructed: &[Gaussian],
    packed: &PackedRasterScene,
) -> RoundtripDiagnostics {
    assert_eq!(
        original_sorted.len(),
        reconstructed.len(),
        "roundtrip diagnostics require equal-length gaussian arrays"
    );

    RoundtripDiagnostics {
        xyz_rmse: [
            rmse_for_channel(original_sorted, reconstructed, |g| g.xyz[0]),
            rmse_for_channel(original_sorted, reconstructed, |g| g.xyz[1]),
            rmse_for_channel(original_sorted, reconstructed, |g| g.xyz[2]),
        ],
        sh_dc_rmse: [
            rmse_for_channel(original_sorted, reconstructed, |g| g.sh_dc[0]),
            rmse_for_channel(original_sorted, reconstructed, |g| g.sh_dc[1]),
            rmse_for_channel(original_sorted, reconstructed, |g| g.sh_dc[2]),
        ],
        original_neighbor_deltas: compute_neighbor_delta_stats(original_sorted),
        reconstructed_neighbor_deltas: compute_neighbor_delta_stats(reconstructed),
        plane_stats: compute_plane_stats(packed),
        byte_split_plane_stats: compute_byte_split_plane_stats(packed),
    }
}

pub fn compute_neighbor_delta_stats(gaussians: &[Gaussian]) -> NeighborDeltaStats {
    NeighborDeltaStats {
        xyz: [
            delta_stats(gaussians, |g| g.xyz[0]),
            delta_stats(gaussians, |g| g.xyz[1]),
            delta_stats(gaussians, |g| g.xyz[2]),
        ],
        sh_dc: [
            delta_stats(gaussians, |g| g.sh_dc[0]),
            delta_stats(gaussians, |g| g.sh_dc[1]),
            delta_stats(gaussians, |g| g.sh_dc[2]),
        ],
    }
}

pub fn sort_gaussians_by_morton(scene: &Scene) -> Vec<Gaussian> {
    let mut sorted = scene.gaussians.clone();
    sorted.sort_unstable_by_key(|g| generate_morton_code(g, &scene.mins, &scene.maxes));
    sorted
}

pub fn compute_plane_stats(packed: &PackedRasterScene) -> Vec<PlaneStats> {
    vec![
        plane_stats_u8("occupancy", &packed.occupancy, &packed.occupancy, packed.width, packed.height),
        plane_stats_u16("geom_x", &packed.geom_x, &packed.occupancy, packed.width, packed.height),
        plane_stats_u16("geom_y", &packed.geom_y, &packed.occupancy, packed.width, packed.height),
        plane_stats_u16("geom_z", &packed.geom_z, &packed.occupancy, packed.width, packed.height),
        plane_stats_u16("sh_dc_r", &packed.sh_dc_r, &packed.occupancy, packed.width, packed.height),
        plane_stats_u16("sh_dc_g", &packed.sh_dc_g, &packed.occupancy, packed.width, packed.height),
        plane_stats_u16("sh_dc_b", &packed.sh_dc_b, &packed.occupancy, packed.width, packed.height),
    ]
}

pub fn split_u16_plane_to_bytes(data: &[u16]) -> (Vec<u8>, Vec<u8>) {
    let mut hi = Vec::with_capacity(data.len());
    let mut lo = Vec::with_capacity(data.len());
    for &value in data {
        let [lo_byte, hi_byte] = value.to_le_bytes();
        hi.push(hi_byte);
        lo.push(lo_byte);
    }
    (hi, lo)
}

pub fn split_packed_u16_planes(packed: &PackedRasterScene) -> Vec<ByteSplitPlane> {
    let (geom_x_hi, geom_x_lo) = split_u16_plane_to_bytes(&packed.geom_x);
    let (geom_y_hi, geom_y_lo) = split_u16_plane_to_bytes(&packed.geom_y);
    let (geom_z_hi, geom_z_lo) = split_u16_plane_to_bytes(&packed.geom_z);
    let (sh_dc_r_hi, sh_dc_r_lo) = split_u16_plane_to_bytes(&packed.sh_dc_r);
    let (sh_dc_g_hi, sh_dc_g_lo) = split_u16_plane_to_bytes(&packed.sh_dc_g);
    let (sh_dc_b_hi, sh_dc_b_lo) = split_u16_plane_to_bytes(&packed.sh_dc_b);

    vec![
        ByteSplitPlane { name: "geom_x", hi: geom_x_hi, lo: geom_x_lo },
        ByteSplitPlane { name: "geom_y", hi: geom_y_hi, lo: geom_y_lo },
        ByteSplitPlane { name: "geom_z", hi: geom_z_hi, lo: geom_z_lo },
        ByteSplitPlane { name: "sh_dc_r", hi: sh_dc_r_hi, lo: sh_dc_r_lo },
        ByteSplitPlane { name: "sh_dc_g", hi: sh_dc_g_hi, lo: sh_dc_g_lo },
        ByteSplitPlane { name: "sh_dc_b", hi: sh_dc_b_hi, lo: sh_dc_b_lo },
    ]
}

pub fn compute_byte_split_plane_stats(packed: &PackedRasterScene) -> Vec<ByteSplitPlaneStats> {
    let byte_split = split_packed_u16_planes(packed);
    let base_stats = compute_plane_stats(packed);

    byte_split
        .into_iter()
        .zip(base_stats.into_iter().skip(1))
        .map(|(plane, base_u16)| ByteSplitPlaneStats {
            name: plane.name,
            base_u16,
            hi: plane_stats_u8(
                match plane.name {
                    "geom_x" => "geom_x_hi",
                    "geom_y" => "geom_y_hi",
                    "geom_z" => "geom_z_hi",
                    "sh_dc_r" => "sh_dc_r_hi",
                    "sh_dc_g" => "sh_dc_g_hi",
                    "sh_dc_b" => "sh_dc_b_hi",
                    _ => "unknown_hi",
                },
                &plane.hi,
                &packed.occupancy,
                packed.width,
                packed.height,
            ),
            lo: plane_stats_u8(
                match plane.name {
                    "geom_x" => "geom_x_lo",
                    "geom_y" => "geom_y_lo",
                    "geom_z" => "geom_z_lo",
                    "sh_dc_r" => "sh_dc_r_lo",
                    "sh_dc_g" => "sh_dc_g_lo",
                    "sh_dc_b" => "sh_dc_b_lo",
                    _ => "unknown_lo",
                },
                &plane.lo,
                &packed.occupancy,
                packed.width,
                packed.height,
            ),
        })
        .collect()
}

fn write_u8_plane(path: impl AsRef<Path>, data: &[u8]) -> std::io::Result<()> {
    let mut writer = BufWriter::new(File::create(path.as_ref())?);
    writer.write_all(data)?;
    writer.flush()
}

fn write_u16_plane(path: impl AsRef<Path>, data: &[u16]) -> std::io::Result<()> {
    let mut writer = BufWriter::new(File::create(path.as_ref())?);
    for value in data {
        writer.write_all(&value.to_le_bytes())?;
    }
    writer.flush()
}

fn read_u8_plane(path: impl AsRef<Path>, expected_len: usize) -> std::io::Result<Vec<u8>> {
    let mut reader = BufReader::new(File::open(path.as_ref())?);
    let mut data = Vec::new();
    reader.read_to_end(&mut data)?;
    if data.len() != expected_len {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("u8 plane length mismatch: expected {expected_len}, got {}", data.len()),
        ));
    }
    Ok(data)
}

fn read_u16_plane(path: impl AsRef<Path>, expected_len: usize) -> std::io::Result<Vec<u16>> {
    let mut reader = BufReader::new(File::open(path.as_ref())?);
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;
    let expected_bytes = expected_len * std::mem::size_of::<u16>();
    if bytes.len() != expected_bytes {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "u16 plane byte length mismatch: expected {expected_bytes}, got {}",
                bytes.len()
            ),
        ));
    }

    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect())
}

fn compute_quant_params(gaussians: &[Gaussian]) -> ([QuantParams; 3], [QuantParams; 3]) {
    let mut geom = [
        QuantParams { min: 0.0, max: 0.0 },
        QuantParams { min: 0.0, max: 0.0 },
        QuantParams { min: 0.0, max: 0.0 },
    ];
    let mut sh_dc = geom;

    if let Some(first) = gaussians.first() {
        geom = [
            QuantParams { min: first.xyz[0], max: first.xyz[0] },
            QuantParams { min: first.xyz[1], max: first.xyz[1] },
            QuantParams { min: first.xyz[2], max: first.xyz[2] },
        ];
        sh_dc = [
            QuantParams { min: first.sh_dc[0], max: first.sh_dc[0] },
            QuantParams { min: first.sh_dc[1], max: first.sh_dc[1] },
            QuantParams { min: first.sh_dc[2], max: first.sh_dc[2] },
        ];
    }

    for gaussian in gaussians {
        update_quant_bounds(&mut geom[0], gaussian.xyz[0]);
        update_quant_bounds(&mut geom[1], gaussian.xyz[1]);
        update_quant_bounds(&mut geom[2], gaussian.xyz[2]);
        update_quant_bounds(&mut sh_dc[0], gaussian.sh_dc[0]);
        update_quant_bounds(&mut sh_dc[1], gaussian.sh_dc[1]);
        update_quant_bounds(&mut sh_dc[2], gaussian.sh_dc[2]);
    }

    (geom, sh_dc)
}

fn update_quant_bounds(params: &mut QuantParams, value: f32) {
    params.min = params.min.min(value);
    params.max = params.max.max(value);
}

fn quantize_to_u16(value: f32, params: QuantParams) -> u16 {
    if params.max <= params.min {
        return 0;
    }

    let normalized = ((value - params.min) / (params.max - params.min)).clamp(0.0, 1.0);
    (normalized * u16::MAX as f32).round() as u16
}

fn dequantize_from_u16(value: u16, params: QuantParams) -> f32 {
    if params.max <= params.min {
        return params.min;
    }

    params.min + (value as f32 / u16::MAX as f32) * (params.max - params.min)
}

fn compute_xyz_bounds(gaussians: &[Gaussian]) -> ([f32; 3], [f32; 3]) {
    let Some(first) = gaussians.first() else {
        return ([0.0; 3], [0.0; 3]);
    };

    let mut mins = first.xyz;
    let mut maxes = first.xyz;
    for gaussian in gaussians.iter().skip(1) {
        for axis in 0..3 {
            mins[axis] = mins[axis].min(gaussian.xyz[axis]);
            maxes[axis] = maxes[axis].max(gaussian.xyz[axis]);
        }
    }

    (mins, maxes)
}

fn rmse_for_channel(
    original: &[Gaussian],
    reconstructed: &[Gaussian],
    accessor: impl Fn(&Gaussian) -> f32,
) -> f64 {
    if original.is_empty() {
        return 0.0;
    }

    let mse = original
        .iter()
        .zip(reconstructed.iter())
        .map(|(lhs, rhs)| {
            let diff = accessor(lhs) as f64 - accessor(rhs) as f64;
            diff * diff
        })
        .sum::<f64>()
        / original.len() as f64;

    mse.sqrt()
}

fn delta_stats(gaussians: &[Gaussian], accessor: impl Fn(&Gaussian) -> f32) -> ChannelDeltaStats {
    if gaussians.len() < 2 {
        return ChannelDeltaStats {
            count: 0,
            mean_abs: 0.0,
            std_abs: 0.0,
        };
    }

    let abs_deltas: Vec<f64> = gaussians
        .windows(2)
        .map(|pair| (accessor(&pair[1]) as f64 - accessor(&pair[0]) as f64).abs())
        .collect();

    let count = abs_deltas.len();
    let mean_abs = abs_deltas.iter().sum::<f64>() / count as f64;
    let variance = abs_deltas
        .iter()
        .map(|delta| {
            let centered = delta - mean_abs;
            centered * centered
        })
        .sum::<f64>()
        / count as f64;

    ChannelDeltaStats {
        count,
        mean_abs,
        std_abs: variance.sqrt(),
    }
}

fn plane_stats_u8(
    name: &'static str,
    plane: &[u8],
    occupancy: &[u8],
    width: u32,
    height: u32,
) -> PlaneStats {
    let values: Vec<u32> = plane
        .iter()
        .zip(occupancy.iter())
        .filter_map(|(&value, &occ)| (occ != 0).then_some(value as u32))
        .collect();
    PlaneStats {
        name,
        min: values.iter().copied().min().unwrap_or(0),
        max: values.iter().copied().max().unwrap_or(0),
        entropy_bits_per_symbol: entropy_bits(&values),
        neighbor_delta: delta_stats_from_u32(&values),
        horizontal_correlation: pixel_correlation_u8(plane, occupancy, width, height, true),
        vertical_correlation: pixel_correlation_u8(plane, occupancy, width, height, false),
    }
}

fn plane_stats_u16(
    name: &'static str,
    plane: &[u16],
    occupancy: &[u8],
    width: u32,
    height: u32,
) -> PlaneStats {
    let values: Vec<u32> = plane
        .iter()
        .zip(occupancy.iter())
        .filter_map(|(&value, &occ)| (occ != 0).then_some(value as u32))
        .collect();
    PlaneStats {
        name,
        min: values.iter().copied().min().unwrap_or(0),
        max: values.iter().copied().max().unwrap_or(0),
        entropy_bits_per_symbol: entropy_bits(&values),
        neighbor_delta: delta_stats_from_u32(&values),
        horizontal_correlation: pixel_correlation_u16(plane, occupancy, width, height, true),
        vertical_correlation: pixel_correlation_u16(plane, occupancy, width, height, false),
    }
}

fn delta_stats_from_u32(values: &[u32]) -> ChannelDeltaStats {
    if values.len() < 2 {
        return ChannelDeltaStats {
            count: 0,
            mean_abs: 0.0,
            std_abs: 0.0,
        };
    }

    let abs_deltas: Vec<f64> = values
        .windows(2)
        .map(|pair| (pair[1] as f64 - pair[0] as f64).abs())
        .collect();
    let count = abs_deltas.len();
    let mean_abs = abs_deltas.iter().sum::<f64>() / count as f64;
    let variance = abs_deltas
        .iter()
        .map(|delta| {
            let centered = delta - mean_abs;
            centered * centered
        })
        .sum::<f64>()
        / count as f64;

    ChannelDeltaStats {
        count,
        mean_abs,
        std_abs: variance.sqrt(),
    }
}

fn entropy_bits(values: &[u32]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mut histogram: HashMap<u32, usize> = HashMap::new();
    for &value in values {
        *histogram.entry(value).or_insert(0) += 1;
    }

    let total = values.len() as f64;
    histogram
        .values()
        .map(|&count| {
            let p = count as f64 / total;
            -p * p.log2()
        })
        .sum()
}

fn pixel_correlation_u8(
    plane: &[u8],
    occupancy: &[u8],
    width: u32,
    height: u32,
    horizontal: bool,
) -> CorrelationStats {
    pixel_correlation(width, height, occupancy, |idx| plane[idx] as f64, horizontal)
}

fn pixel_correlation_u16(
    plane: &[u16],
    occupancy: &[u8],
    width: u32,
    height: u32,
    horizontal: bool,
) -> CorrelationStats {
    pixel_correlation(width, height, occupancy, |idx| plane[idx] as f64, horizontal)
}

fn pixel_correlation(
    width: u32,
    height: u32,
    occupancy: &[u8],
    value_at: impl Fn(usize) -> f64,
    horizontal: bool,
) -> CorrelationStats {
    let width = width as usize;
    let height = height as usize;
    let mut xs = Vec::new();
    let mut ys = Vec::new();

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let neighbor = if horizontal {
                (x + 1 < width).then_some(idx + 1)
            } else {
                (y + 1 < height).then_some(idx + width)
            };
            let Some(neighbor_idx) = neighbor else {
                continue;
            };
            if occupancy[idx] == 0 || occupancy[neighbor_idx] == 0 {
                continue;
            }
            xs.push(value_at(idx));
            ys.push(value_at(neighbor_idx));
        }
    }

    pearson_from_pairs(&xs, &ys)
}

fn pearson_from_pairs(xs: &[f64], ys: &[f64]) -> CorrelationStats {
    assert_eq!(xs.len(), ys.len(), "correlation inputs must match in length");
    if xs.len() < 2 {
        return CorrelationStats {
            count: xs.len(),
            pearson_r: 0.0,
        };
    }

    let count = xs.len();
    let mean_x = xs.iter().sum::<f64>() / count as f64;
    let mean_y = ys.iter().sum::<f64>() / count as f64;
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        let dx = x - mean_x;
        let dy = y - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    let pearson_r = if denom > 0.0 { cov / denom } else { 0.0 };
    CorrelationStats { count, pearson_r }
}

fn write_preview_u8_png(
    path: impl AsRef<Path>,
    width: u32,
    height: u32,
    data: &[u8],
    occupancy: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut pixels = vec![0u8; data.len()];
    for (idx, &value) in data.iter().enumerate() {
        pixels[idx] = if occupancy[idx] != 0 { value.saturating_mul(255) } else { 0 };
    }
    write_grayscale_png(path, width, height, &pixels)
}

fn write_preview_u16_png(
    path: impl AsRef<Path>,
    width: u32,
    height: u32,
    data: &[u16],
    occupancy: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    let occupied_values: Vec<u16> = data
        .iter()
        .zip(occupancy.iter())
        .filter_map(|(&value, &occ)| (occ != 0).then_some(value))
        .collect();
    let min = occupied_values.iter().copied().min().unwrap_or(0);
    let max = occupied_values.iter().copied().max().unwrap_or(min);
    let range = max.saturating_sub(min);

    let mut pixels = vec![0u8; data.len()];
    for (idx, &value) in data.iter().enumerate() {
        if occupancy[idx] == 0 {
            continue;
        }
        pixels[idx] = if range == 0 {
            0
        } else {
            (((value.saturating_sub(min)) as f64 / range as f64) * 255.0).round() as u8
        };
    }

    write_grayscale_png(path, width, height, &pixels)
}

fn write_grayscale_png(
    path: impl AsRef<Path>,
    width: u32,
    height: u32,
    pixels: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(path.as_ref())?;
    let writer = BufWriter::new(file);
    let mut encoder = png::Encoder::new(writer, width, height);
    encoder.set_color(png::ColorType::Grayscale);
    encoder.set_depth(png::BitDepth::Eight);
    let mut png_writer = encoder.write_header()?;
    png_writer.write_image_data(pixels)?;
    Ok(())
}

fn write_lossless_u8_png(
    path: impl AsRef<Path>,
    width: u32,
    height: u32,
    pixels: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    write_grayscale_png(path, width, height, pixels)
}

fn write_rgb_png(
    path: impl AsRef<Path>,
    width: u32,
    height: u32,
    pixels: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(path.as_ref())?;
    let writer = BufWriter::new(file);
    let mut encoder = png::Encoder::new(writer, width, height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut png_writer = encoder.write_header()?;
    png_writer.write_image_data(pixels)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_scene(num_gaussians: usize) -> Scene {
        let mut gaussians = Vec::with_capacity(num_gaussians);
        for i in 0..num_gaussians {
            let i_f = i as f32;
            gaussians.push(Gaussian {
                xyz: [
                    (i_f * 0.71).sin() * 8.0 + 3.0,
                    (i_f * 0.37).cos() * 5.0 - 1.5,
                    i_f * 0.125 + (i_f * 0.19).sin(),
                ],
                normals: None,
                sh_dc: [
                    (i_f * 0.11).sin() * 0.7,
                    (i_f * 0.07).cos() * 0.5,
                    ((i_f * 0.17).sin() + (i_f * 0.13).cos()) * 0.25,
                ],
                sh_rest: vec![0.0; 45],
                opacity: 0.5,
                scale: [0.0; 3],
                rot: [0.0, 0.0, 0.0, 1.0],
            });
        }

        let (mins, maxes) = compute_xyz_bounds(&gaussians);
        let mut meta = HashMap::new();
        meta.insert("format".to_string(), "synthetic".to_string());
        meta.insert("vertex_count".to_string(), gaussians.len().to_string());

        Scene {
            gaussians,
            meta,
            mins,
            maxes,
        }
    }

    #[test]
    fn pack_scene_creates_expected_raster_shape() {
        let scene = synthetic_scene(17);
        let packed = pack_scene_to_raster(&scene, 4);

        assert_eq!(packed.num_gaussians, 17);
        assert_eq!(packed.width, 4);
        assert_eq!(packed.height, 5);
        assert_eq!(packed.occupancy.len(), 20);
        assert_eq!(packed.occupancy.iter().filter(|&&v| v == 1).count(), 17);
        assert_eq!(packed.sh_rest_len, 45);
    }

    #[test]
    fn pack_unpack_roundtrip_preserves_xyz_and_sh_dc_with_low_error() {
        let scene = synthetic_scene(64);
        let sorted = sort_gaussians_by_morton(&scene);
        let packed = pack_scene_to_raster(&scene, 8);
        let reconstructed = unpack_raster_to_scene(&packed);
        let diagnostics = compute_roundtrip_diagnostics(&sorted, &reconstructed.gaussians, &packed);

        for axis in diagnostics.xyz_rmse {
            assert!(axis < 0.0002, "xyz RMSE too high: {axis}");
        }
        for axis in diagnostics.sh_dc_rmse {
            assert!(axis < 0.00005, "sh_dc RMSE too high: {axis}");
        }

        assert_eq!(reconstructed.gaussians.len(), sorted.len());
        assert!(reconstructed
            .gaussians
            .iter()
            .all(|g| g.sh_rest.len() == 45 && g.opacity == 1.0 && g.rot == [0.0, 0.0, 0.0, 1.0]));
    }

    #[test]
    fn neighbor_delta_stats_match_expected_count() {
        let scene = synthetic_scene(32);
        let sorted = sort_gaussians_by_morton(&scene);
        let stats = compute_neighbor_delta_stats(&sorted);

        for channel in stats.xyz {
            assert_eq!(channel.count, 31);
            assert!(channel.mean_abs >= 0.0);
            assert!(channel.std_abs >= 0.0);
        }
        for channel in stats.sh_dc {
            assert_eq!(channel.count, 31);
            assert!(channel.mean_abs >= 0.0);
            assert!(channel.std_abs >= 0.0);
        }
    }

    #[test]
    fn empty_scene_roundtrip_is_supported() {
        let scene = synthetic_scene(0);
        let packed = pack_scene_to_raster(&scene, 16);
        let reconstructed = unpack_raster_to_scene(&packed);

        assert_eq!(packed.height, 0);
        assert!(packed.occupancy.is_empty());
        assert!(reconstructed.gaussians.is_empty());
        assert_eq!(reconstructed.mins, [0.0; 3]);
        assert_eq!(reconstructed.maxes, [0.0; 3]);
    }

    #[test]
    fn directory_container_roundtrip_preserves_packed_scene() {
        let scene = synthetic_scene(25);
        let packed = pack_scene_to_raster(&scene, 6);
        let unique = format!(
            "gaussian_packing_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let dir = std::env::temp_dir().join(unique);

        write_packed_raster_scene_dir(&packed, &dir).unwrap();
        let loaded = read_packed_raster_scene_dir(&dir).unwrap();

        assert_eq!(loaded, packed);

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn plane_stats_are_generated_for_all_planes() {
        let scene = synthetic_scene(12);
        let packed = pack_scene_to_raster(&scene, 4);
        let stats = compute_plane_stats(&packed);

        assert_eq!(stats.len(), 7);
        assert_eq!(stats[0].name, "occupancy");
        assert!(stats.iter().all(|plane| plane.max >= plane.min));
    }

    #[test]
    fn byte_split_planes_and_stats_are_generated() {
        let scene = synthetic_scene(10);
        let packed = pack_scene_to_raster(&scene, 4);
        let split = split_packed_u16_planes(&packed);
        let stats = compute_byte_split_plane_stats(&packed);

        assert_eq!(split.len(), 6);
        assert_eq!(stats.len(), 6);
        assert_eq!(split[0].hi.len(), packed.geom_x.len());
        assert_eq!(split[0].lo.len(), packed.geom_x.len());
        assert_eq!(stats[0].name, "geom_x");
        assert!(stats.iter().all(|entry| entry.base_u16.max >= entry.base_u16.min));
    }

    #[test]
    fn geom_hi_png_roundtrip_rebuilds_original_geom_planes() {
        let scene = synthetic_scene(18);
        let packed = pack_scene_to_raster(&scene, 5);
        let sorted = sort_gaussians_by_morton(&scene);
        let unique = format!(
            "gaussian_packing_geom_hi_png_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let dir = std::env::temp_dir().join(unique);

        write_geom_hi_pngs(&packed, &dir).unwrap();
        let decoded_hi = load_geom_hi_pngs(&dir, packed.width, packed.height).unwrap();
        let rebuilt = rebuild_packed_scene_with_decoded_geom_hi(
            &packed,
            [&decoded_hi[0], &decoded_hi[1], &decoded_hi[2]],
        );
        assert_eq!(rebuilt.geom_x, packed.geom_x);
        assert_eq!(rebuilt.geom_y, packed.geom_y);
        assert_eq!(rebuilt.geom_z, packed.geom_z);

        let result = evaluate_geom_hi_png_roundtrip(&sorted, &packed, &dir).unwrap();
        for axis in result.xyz_rmse {
            assert!(axis < 0.0002, "geometry RMSE too high after hi-byte png roundtrip: {axis}");
        }

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn geom_hi_rgb_png_roundtrip_rebuilds_original_geom_planes() {
        let scene = synthetic_scene(18);
        let packed = pack_scene_to_raster(&scene, 5);
        let sorted = sort_gaussians_by_morton(&scene);
        let unique = format!(
            "gaussian_packing_geom_hi_rgb_png_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let dir = std::env::temp_dir().join(unique);

        write_geom_hi_rgb_pngs_helper(&packed, &dir).unwrap();
        let decoded_hi = load_geom_hi_rgb_png(&dir, packed.width, packed.height).unwrap();
        let rebuilt = rebuild_packed_scene_with_decoded_geom_hi(
            &packed,
            [&decoded_hi[0], &decoded_hi[1], &decoded_hi[2]],
        );
        assert_eq!(rebuilt.geom_x, packed.geom_x);
        assert_eq!(rebuilt.geom_y, packed.geom_y);
        assert_eq!(rebuilt.geom_z, packed.geom_z);

        let result = evaluate_geom_hi_rgb_png_roundtrip(&sorted, &packed, &dir).unwrap();
        for axis in result.xyz_rmse {
            assert!(axis < 0.0002, "geometry RMSE too high after rgb hi-byte png roundtrip: {axis}");
        }

        fs::remove_dir_all(dir).unwrap();
    }

    fn write_geom_hi_rgb_pngs_helper(
        packed: &PackedRasterScene,
        dir: &Path,
    ) -> Result<GeomHiRgbPngPath, Box<dyn std::error::Error>> {
        write_geom_hi_rgb_png(packed, dir)
    }
}
