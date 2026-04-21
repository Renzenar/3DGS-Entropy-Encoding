use crate::frame::Frame;

pub fn downscale_frame(input: &Frame, factor: u32) -> Frame {
    let f = factor as usize;
    let new_w = input.width / f;
    let new_h = input.height / f;

    let mut out = Frame::new(new_w, new_h);

    for y in 0..new_h {
        for x in 0..new_w {
            let mut sum = 0.0;

            for dy in 0..f {
                for dx in 0..f {
                    sum += input.get(x * f + dx, y * f + dy);
                }
            }

            out.set(x, y, sum / (f * f) as f32);
        }
    }

    out
}

pub fn upscale_frame(input: &Frame, factor: u32) -> Frame {
    let f = factor as usize;
    let new_w = input.width * f;
    let new_h = input.height * f;

    let mut out = Frame::new(new_w, new_h);

    for y in 0..new_h {
        for x in 0..new_w {
            let src_x = x / f;
            let src_y = y / f;
            out.set(x, y, input.get(src_x, src_y));
        }
    }

    out
}
