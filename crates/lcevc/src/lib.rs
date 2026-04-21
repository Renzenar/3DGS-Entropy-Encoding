pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}

use video::frame::Frame;

/// Quantize residual → i16 → bytes
/// Quantize residual → i16 → bytes
pub fn encode_lcevc_residual(residual: &Frame) -> Vec<u8> {
    let mut out = Vec::with_capacity(residual.data.len() * 2);

    for v in residual.data.iter() {
        let v: f32 = *v;

        let q = (v * 32767.0).round() as i16;
        out.extend_from_slice(&q.to_le_bytes());
    }

    out
}

/// Decode residual
pub fn decode_lcevc_residual(bytes: &[u8], width: usize, height: usize) -> Frame {
    let mut frame = Frame::new(width, height);

    for i in 0..frame.data.len() {
        let b0 = bytes[2 * i];
        let b1 = bytes[2 * i + 1];
        let val = i16::from_le_bytes([b0, b1]);
        frame.data[i] = val as f32 / 32767.0;
    }

    frame
}
