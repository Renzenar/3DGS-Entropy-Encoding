//! eparser: Kerbl-style 3DGS PLY parser (binary LE + ASCII fallback), no deps.
//!
//! Schema (Kerbl):
//! x y z | (optional) nx ny nz | f_dc_0..2 | f_rest_0..44 | opacity | scale_0..2 | rot_0..3

use std::collections::HashMap;
use std::error::Error as StdError;
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read, Seek, SeekFrom};

#[derive(Debug, Clone)]
pub struct Gaussians {
    pub xyz: Vec<[f32; 3]>,
    pub normals: Option<Vec<[f32; 3]>>,
    pub sh_dc: Vec<[f32; 3]>,
    pub sh_rest: Vec<Vec<f32>>, // uniform length per point (e.g., 45)
    pub opacity: Vec<f32>,
    pub scale: Vec<[f32; 3]>,   // log-scales
    pub rot: Vec<[f32; 4]>,     // quaternion xyzw (normalized)
    pub meta: HashMap<String, String>,
}

#[derive(Debug)]
pub enum PlyError {
    Io(io::Error),
    Header(&'static str),
    Eof,
    MissingProp(&'static str),
    UnsupportedFormat(String),
    BadNumber,
}
impl Display for PlyError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            PlyError::Io(e) => write!(f, "io error: {e}"),
            PlyError::Header(s) => write!(f, "invalid/unsupported PLY header: {s}"),
            PlyError::Eof => write!(f, "unexpected EOF while reading vertex data"),
            PlyError::MissingProp(s) => write!(f, "required property missing: {s}"),
            PlyError::UnsupportedFormat(s) => write!(f, "unsupported PLY format: {s}"),
            PlyError::BadNumber => write!(f, "invalid numeric field"),
        }
    }
}
impl StdError for PlyError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        if let PlyError::Io(e) = self { Some(e) } else { None }
    }
}
impl From<io::Error> for PlyError { fn from(e: io::Error) -> Self { PlyError::Io(e) } }

#[derive(Debug)]
struct PlyMeta {
    format: PlyFormat,
    vertex_count: usize,
    properties: Vec<(String, String)>, // (name, dtype)
    header_len: u64,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlyFormat { BinaryLittleEndian, Ascii }
impl PlyFormat {
    fn from_header_line(line: &str) -> Result<Self, PlyError> {
        if line.contains("binary_little_endian") { Ok(Self::BinaryLittleEndian) }
        else if line.contains("ascii") { Ok(Self::Ascii) }
        else { Err(PlyError::UnsupportedFormat(line.to_string())) }
    }
}

/// Load a Kerbl-style 3DGS PLY file.
pub fn load_gaussians_from_ply(path: &str) -> Result<Gaussians, PlyError> {
    let file = File::open(path)?;
    let mut br = BufReader::new(file);
    let meta = parse_header(&mut br)?;

    let rows: Vec<Vec<f32>> = match meta.format {
        PlyFormat::BinaryLittleEndian => {
            let mut f = br.into_inner();
            read_vertices_binary(&mut f, &meta)?
        }
        PlyFormat::Ascii => read_vertices_ascii(&mut br, &meta)?,
    };
    let names: Vec<String> = meta.properties.iter().map(|(n, _)| n.clone()).collect();

    // Required xyz
    let x = grab_col(&names, &rows, &["x"])?;
    let y = grab_col(&names, &rows, &["y"])?;
    let z = grab_col(&names, &rows, &["z"])?;
    let mut xyz = Vec::with_capacity(rows.len());
    for i in 0..rows.len() { xyz.push([x[i], y[i], z[i]]); }

    // Optional normals
    let normals = grab_triple_optional(&names, &rows, "nx", "ny", "nz");

    // opacity
    let opacity = grab_col(&names, &rows, &["opacity", "alpha"])?;

    // scale_0..2
    let scale = grab_prefix_triple(&names, &rows, &["scale_", "scale", "s", "sc"])?;

    // rot_0..3 or qx..qw
    let rot = if ["rot_0","rot_1","rot_2","rot_3"].iter().all(|k| names.contains(&k.to_string())) {
        let i0 = idx(&names, "rot_0")?; let i1 = idx(&names, "rot_1")?;
        let i2 = idx(&names, "rot_2")?; let i3 = idx(&names, "rot_3")?;
        rows.iter().map(|r| [r[i0], r[i1], r[i2], r[i3]]).collect()
    } else if ["qx","qy","qz","qw"].iter().all(|k| names.contains(&k.to_string())) {
        let ix = idx(&names, "qx")?; let iy = idx(&names, "qy")?;
        let iz = idx(&names, "qz")?; let iw = idx(&names, "qw")?;
        rows.iter().map(|r| [r[ix], r[iy], r[iz], r[iw]]).collect()
    } else {
        return Err(PlyError::MissingProp("rot quaternion"));
    };
    let rot: Vec<[f32; 4]> = rot.into_iter().map(|q| {
        let n = (q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]).sqrt().max(1e-8);
        [q[0]/n, q[1]/n, q[2]/n, q[3]/n]
    }).collect();

    // SH DC
    let dc0 = grab_col(&names, &rows, &["f_dc_0","sh_dc_0","dc_0"])?;
    let dc1 = grab_col(&names, &rows, &["f_dc_1","sh_dc_1","dc_1"])?;
    let dc2 = grab_col(&names, &rows, &["f_dc_2","sh_dc_2","dc_2"])?;
    let mut sh_dc = Vec::with_capacity(rows.len());
    for i in 0..rows.len() { sh_dc.push([dc0[i], dc1[i], dc2[i]]); }

    // SH rest (prefer f_rest_*)
    let mut rest_cols: Vec<(usize, usize)> = Vec::new();
    for (i, n) in names.iter().enumerate() {
        if let Some(sfx) = n.strip_prefix("f_rest_")
            .or_else(|| n.strip_prefix("sh_rest_"))
            .or_else(|| n.strip_prefix("rest_")) {
            if let Ok(k) = sfx.parse::<usize>() { rest_cols.push((i, k)); }
        }
    }
    rest_cols.sort_by_key(|&(_, k)| k);
    let sh_rest: Vec<Vec<f32>> = if !rest_cols.is_empty() {
        rows.iter().map(|r| rest_cols.iter().map(|(i, _)| r[*i]).collect()).collect()
    } else {
        // fallback: f_0.., first 3 are DC
        let mut fcols: Vec<(usize, usize)> = Vec::new();
        for (i, n) in names.iter().enumerate() {
            if let Some(k) = n.strip_prefix("f_")
                .or_else(|| n.strip_prefix("sh_"))
                .and_then(|s| s.parse::<usize>().ok()) {
                fcols.push((i, k));
            }
        }
        fcols.sort_by_key(|&(_, k)| k);
        if fcols.len() >= 3 {
            let rest_idxs: Vec<usize> = fcols.into_iter().filter(|&(_, k)| k >= 3).map(|(i, _)| i).collect();
            rows.iter().map(|r| rest_idxs.iter().map(|i| r[*i]).collect()).collect()
        } else {
            vec![Vec::new(); rows.len()]
        }
    };

    let mut meta_map = HashMap::new();
    meta_map.insert(
        "format".into(),
        match meta.format { PlyFormat::BinaryLittleEndian => "binary_little_endian", PlyFormat::Ascii => "ascii" }.to_string(),
    );
    meta_map.insert("vertex_count".into(), meta.vertex_count.to_string());

    Ok(Gaussians { xyz, normals, sh_dc, sh_rest, opacity, scale, rot, meta: meta_map })
}

// ---------------- internal helpers ----------------

fn parse_header(r: &mut BufReader<File>) -> Result<PlyMeta, PlyError> {
    let mut header_len: u64 = 0;
    let mut header: Vec<String> = Vec::new();
    loop {
        let mut line = String::new();
        let n = r.read_line(&mut line)?;
        if n == 0 { return Err(PlyError::Header("EOF in header")); }
        header_len += n as u64;
        let trimmed = line.trim_end().to_string();
        if header.is_empty() && !trimmed.starts_with("ply") { return Err(PlyError::Header("missing 'ply' signature")); }
        header.push(trimmed);
        if header.last().unwrap() == "end_header" { break; }
    }
    let fmt_line = header.iter().find(|l| l.starts_with("format ")).ok_or(PlyError::Header("missing 'format'"))?;
    let format = PlyFormat::from_header_line(fmt_line)?;

    let mut vertex_count: Option<usize> = None;
    let mut in_vertex = false;
    let mut properties: Vec<(String, String)> = Vec::new();
    for l in &header {
        if l.starts_with("element ") {
            in_vertex = l.starts_with("element vertex");
            if in_vertex {
                let parts: Vec<&str> = l.split_whitespace().collect();
                let count = parts.last().ok_or(PlyError::Header("bad 'element vertex'"))?
                    .parse::<usize>().map_err(|_| PlyError::Header("bad vertex count"))?;
                vertex_count = Some(count);
            }
        } else if in_vertex && l.starts_with("property ") {
            let parts: Vec<&str> = l.split_whitespace().collect();
            if parts.get(1) == Some(&"list") { continue; } // skip lists
            let dtype = parts.get(1).ok_or(PlyError::Header("bad property dtype"))?;
            let name  = parts.get(2).ok_or(PlyError::Header("bad property name"))?;
            properties.push(((*name).to_string(), (*dtype).to_string()));
        }
    }
    Ok(PlyMeta {
        format,
        vertex_count: vertex_count.ok_or(PlyError::Header("no vertex element"))?,
        properties,
        header_len,
    })
}

fn dtype_size(t: &str) -> usize {
    match t {
        "char" | "uchar" => 1,
        "short" | "ushort" => 2,
        "int" | "uint" | "float" => 4,
        "double" => 8,
        _ => 4,
    }
}

fn read_vertices_binary(file: &mut File, meta: &PlyMeta) -> Result<Vec<Vec<f32>>, PlyError> {
    file.seek(SeekFrom::Start(meta.header_len))?;
    let stride: usize = meta.properties.iter().map(|(_, dt)| dtype_size(dt)).sum();
    let mut buf = vec![0u8; meta.vertex_count * stride];
    file.read_exact(&mut buf).map_err(|_| PlyError::Eof)?;

    let mut rows: Vec<Vec<f32>> = Vec::with_capacity(meta.vertex_count);
    let mut offset = 0usize;
    for _ in 0..meta.vertex_count {
        let mut row: Vec<f32> = Vec::with_capacity(meta.properties.len());
        let mut off = offset;
        for (_, dt) in &meta.properties {
            let f = match dt.as_str() {
                "char"   => { let v = buf[off] as i8;  off+=1; v as f32 }
                "uchar"  => { let v = buf[off] as u8;  off+=1; v as f32 }
                "short"  => { let v = i16::from_le_bytes(buf[off..off+2].try_into().unwrap()); off+=2; v as f32 }
                "ushort" => { let v = u16::from_le_bytes(buf[off..off+2].try_into().unwrap()); off+=2; v as f32 }
                "int"    => { let v = i32::from_le_bytes(buf[off..off+4].try_into().unwrap()); off+=4; v as f32 }
                "uint"   => { let v = u32::from_le_bytes(buf[off..off+4].try_into().unwrap()); off+=4; v as f32 }
                "double" => { let v = f64::from_le_bytes(buf[off..off+8].try_into().unwrap()); off+=8; v as f32 }
                _        => { let v = f32::from_le_bytes(buf[off..off+4].try_into().unwrap()); off+=4; v }
            };
            row.push(f);
        }
        rows.push(row);
        offset += stride;
    }
    Ok(rows)
}

fn read_vertices_ascii(reader: &mut BufReader<File>, meta: &PlyMeta) -> Result<Vec<Vec<f32>>, PlyError> {
    let mut rows: Vec<Vec<f32>> = Vec::with_capacity(meta.vertex_count);
    for _ in 0..meta.vertex_count {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        let mut vals: Vec<f32> = Vec::with_capacity(meta.properties.len());
        for tok in line.split_whitespace().take(meta.properties.len()) {
            vals.push(tok.parse::<f32>().map_err(|_| PlyError::BadNumber)?);
        }
        if vals.len() != meta.properties.len() { vals.resize(meta.properties.len(), 0.0); }
        rows.push(vals);
    }
    Ok(rows)
}

fn idx(names: &[String], needle: &str) -> Result<usize, PlyError> {
    names.iter().position(|n| n == needle).ok_or(PlyError::MissingProp(needle))
}

fn grab_col(names: &[String], rows: &[Vec<f32>], candidates: &[&str]) -> Result<Vec<f32>, PlyError> {
    for c in candidates {
        if let Some(i) = names.iter().position(|n| n == c) {
            return Ok(rows.iter().map(|r| r[i]).collect());
        }
    }
    Err(PlyError::MissingProp(candidates.first().copied().unwrap_or("column")))
}

fn grab_triple_optional(
    names: &[String],
    rows: &[Vec<f32>],
    a: &str, b: &str, c: &str
) -> Option<Vec<[f32; 3]>> {
    let (ia, ib, ic) = (
        names.iter().position(|n| n == a),
        names.iter().position(|n| n == b),
        names.iter().position(|n| n == c),
    );
    match (ia, ib, ic) {
        (Some(ia), Some(ib), Some(ic)) => {
            let mut out = Vec::with_capacity(rows.len());
            for r in rows { out.push([r[ia], r[ib], r[ic]]); }
            Some(out)
        }
        _ => None,
    }
}

fn grab_prefix_triple(
    names: &[String],
    rows: &[Vec<f32>],
    prefixes: &[&str],
) -> Result<Vec<[f32; 3]>, PlyError> {
    for pref in prefixes {
        let form1 = |i: usize| format!("{pref}{i}");
        let base = pref.trim_end_matches('_');
        let form2 = |i: usize| format!("{base}_{i}");
        let mut idxs: [Option<usize>; 3] = [None, None, None];
        for i in 0..3 { idxs[i] = names.iter().position(|n| n == &form1(i) || n == &form2(i)); }
        if idxs.iter().all(|o| o.is_some()) {
            let (i0, i1, i2) = (idxs[0].unwrap(), idxs[1].unwrap(), idxs[2].unwrap());
            let mut out = Vec::with_capacity(rows.len());
            for r in rows { out.push([r[i0], r[i1], r[i2]]); }
            return Ok(out);
        }
    }
    Err(PlyError::MissingProp("scale_0..2"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn quaternion_normalization() {
        let q = [0.0f32, 0.0, 0.0, 10.0];
        let n = (q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]).sqrt().max(1e-8);
        let qn = [q[0]/n, q[1]/n, q[2]/n, q[3]/n];
        assert!((qn[3] - 1.0).abs() < 1e-6);
    }
}
