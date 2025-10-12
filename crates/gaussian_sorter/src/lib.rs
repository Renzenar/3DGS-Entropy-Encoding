//! This will be a morton order sorter
//! The difficulty will be quantizing to an integer for morton ordering while maintaining enough precision to exploit redundancy due to spatial proximity.




//ChatGPT recommended function:
// a_q = clamp( floor( ((a - a_min) / (a_max - a_min)) * (2^b - 1) + 0.5 ), 0, 2^b - 1 )
//
// where:
//   a      = original floating-point coordinate (x, y, or z)
//   a_min  = minimum value of that axis in the dataset
//   a_max  = maximum value of that axis in the dataset
//   b      = number of bits per axis (e.g. 21)
//   a_q    = quantized integer coordinate in [0, 2^b - 1]
//
//notes:
// - this requires knowing the min and max of each value
//    -> we can track min/max during the parsing process without adding much computation
// - this is uniform distribution across min max
//    -> we should explore better quantization approaches to get more accurate predictions
fn quantize(value: f32, min : f32, max : f32, bits: u32) -> u32 {
    let val= 0;

    val
}