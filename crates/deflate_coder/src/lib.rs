// src/main.rs
use libflate::deflate::{Decoder, Encoder};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Payload {
    data: Vec<i32>,
}

pub fn compress_i32_vec(input: Vec<i32>) -> anyhow::Result<Vec<u8>> {
    // 1) Serialize to bytes (endian-stable)
    let payload = Payload { data: input };
    let serialized = bincode::serialize(&payload)?; // Vec<u8>

    // 2) LZ77-based DEFLATE compression
    let mut encoder = Encoder::new(Vec::new());
    encoder.write_all(&serialized)?;
    let compressed = encoder.finish().into_result()?; // Vec<u8>
    Ok(compressed)
}

pub fn decompress_i32_vec(compressed: &[u8]) -> anyhow::Result<Vec<i32>> {
    // 3) DEFLATE decompression
    let mut decoder = Decoder::new(compressed);
    let mut decompressed = Vec::new();
    decoder.read_to_end(&mut decompressed)?;

    // 4) Deserialize back to Vec<i32>
    let restored: Payload = bincode::deserialize(&decompressed)?;
    Ok(restored.data)
}

// fn main() -> anyhow::Result<()> {
//     let original = vec![1, -2, 3, 3, 3, 4, 4, 4, 4, 5];
//     let compressed = compress_i32_vec(original.clone())?;
//     let roundtrip = decompress_i32_vec(&compressed)?;
//
//     println!("original:  {:?}", original);
//     println!("compressed len: {}", compressed.len());
//     println!("roundtrip: {:?}", roundtrip);
//     Ok(())
// }
