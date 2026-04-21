use crate::frame::Frame;

pub fn compute_residual(original: &Frame, approx: &Frame) -> Frame {
    let mut out = Frame::new(original.width, original.height);

    for i in 0..original.data.len() {
        out.data[i] = original.data[i] - approx.data[i];
    }

    out
}

pub fn apply_residual(base: &Frame, residual: &Frame) -> Frame {
    let mut out = Frame::new(base.width, base.height);

    for i in 0..base.data.len() {
        out.data[i] = base.data[i] + residual.data[i];
    }

    out
}
