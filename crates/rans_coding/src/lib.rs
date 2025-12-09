/**
 * File Name: rans_coding
 * Description: This file implements a block-adaptive rANS coder and decoder
 * Date Created: 11/05/2025
 * Date Last Modified: 11/05/2025
 */

use rans::b64_encoder::{B64RansEncSymbol, B64RansEncoderMulti};
use rans::{RansEncSymbol, RansEncoderMulti, RansDecSymbol, RansDecoderMulti};
use rans::b64_decoder::{B64RansDecSymbol, B64RansDecoderMulti};
use bv::BitVec;

const SIGN_ALPH_SIZE : usize = 2;
const EXP_ALPH_SIZE : usize = 1 << 8;
const SCALE_BIT: u32 = 16;
// const MAX_ERR : i32 = 15_000;
const MAX_ERR : i32 = 100;
const STEP : f32 = (1 << 8) as f32;
const MANT_ALPH_SIZE: usize = ((1 << 23) / STEP as usize) + 1;



fn quantize(vals: &mut Vec<f32>) {

    vals.iter_mut().for_each(| val| {
        let mut mant = val.to_bits() & 0x7F_FFFF;
        mant = ((mant as f32 / STEP).round() as u32 * STEP as u32) & 0x7F_FFFF;
        *val = f32::from_bits((val.to_bits() & 0xFF80_0000) | mant);
    });

}
fn second_order_delta_encode(v: &mut [f32]) {
    if v.len() < 3 { return; }
    for i in (2..v.len()).rev() {
        v[i] = v[i] - (v[i-1] + (v[i-1] - v[i-2]));
    }
}

fn second_order_delta_decode(v: &mut [f32]) {
    if v.len() < 3 { return; }
    for i in 2..v.len() {
        v[i] = v[i] + (v[i-1] + (v[i-1] - v[i-2]));
    }
}


struct Context {
    freq: Vec<u16>,
    total_freq: usize,
}

/**
 * Description: Context implements the underlying adaptive context model used by both the encoder
 * and decoder. This model is defined to keep raw counts of each symbol in the given alphabet. The
 * encoder and decoder are responsible for keeping normalized frequency tables for coding.
 * 
 */
impl Context {
    //initialize array to alphabet length + 1 to allow for escape symbol
    //the last index [length] will be the escape symbol
    //NOTE: alphabet_len must be factor of 2 (I should probably enforce this somehow)
    pub fn new(alphabet_len: usize) -> Self {
        Self{
            freq: vec![1; alphabet_len],
            total_freq: alphabet_len,
            }
    }

    pub fn increment_freq(&mut self, idx: usize) {
            self.freq[idx] += 1;
            self.total_freq += 1;
    }

    pub fn get_freq_array(&self) -> &[u16] {
       &self.freq
    }

    pub fn rebuild_histogram(&self) -> bool {
       self.total_freq == (1 << SCALE_BIT)
    }


    /**Description: rescale_model scales each index in the array by a factor of 2. If it scales to
     * 0, it will resolve to 1. Total_freq is updated with the new scale.
     */
    pub fn rescale_model(&mut self) {
        let mut total = 0u32;
        self.freq.iter_mut().for_each(|idx| {
            *idx >>= 1;
            if *idx == 0 {
                *idx = 1;
            }
            total += *idx as u32;
        });
        self.total_freq = total as usize;
    }
}
struct RansEncContext {
    context: Context,
    snapshots: Vec<Vec<B64RansEncSymbol>>,
    //consider turning this into a single flat index, for now this is proof of concept
    //would probably have to communicate how many partitions there are with each data stream
    rescale_location: Vec<usize>,
}

/**
 * Description: RansEnc implements the rans encoder and all necessary methods. 
 * The encoder performs two passes
 * Forward Pass:
 *  - tallies raw counts and rebuilds a normalized frequency table whenever the context model's
 *    total_frequency is equal to 2^SCALE_BIT
 *  - caches previous normalized context model in snapshots to simulate decoder's LIFO forward
 *    processing
 *  - rescales context model at each normalization
 *
 * Backward Pass:
 *  - encodes values in reverse order due to rANS's LIFO nature
 *  - uses the cached histogram from the forward pass at each stage
 *
 */
impl RansEncContext {
    pub fn new(alphabet_len: usize) -> Self {
        let context = Context::new(alphabet_len);
        Self { context, snapshots: vec![], rescale_location: vec![] }
    }

    //might want to add this into component breakdown
    pub fn increment_freq(&mut self, val: u32, idx: usize, channel: usize){
        self.context.increment_freq(val as usize);
        if self.context.rebuild_histogram() {
            self.build_snapshot();
            self.context.rescale_model();
            self.rescale_location.push(idx);
        }
    }

    pub fn build_snapshot(&mut self) {
        let res : Vec<B64RansEncSymbol> = self.context.get_freq_array().iter()
            .scan(0u32, |acc, &x|{
                let val = *acc;
                *acc += x as u32;
                Some((val, x))
            }).map(|(cum_freq, freq)|
            B64RansEncSymbol::new(
                cum_freq,
                freq as u32,
                SCALE_BIT
            )
        ).collect();

        self.snapshots.push(res);
    }

}
const ENC_SIGN_CHANNEL: usize = 0;
const ENC_EXPONENT_CHANNEL: usize = 1;
const ENC_MANTISSA_CHANNEL: usize = 2;
pub struct RansEnc<'a> {
    encode: &'a Vec<f32>,
    sign_context: RansEncContext,
    exponent_context: RansEncContext,
    mantissa_context: RansEncContext,
    encoder: B64RansEncoderMulti<3>
}


impl<'a> RansEnc<'a> {
    pub fn new(buffer_size: usize, encode: &'a Vec<f32>) -> Self {
        let encoder = B64RansEncoderMulti::new(buffer_size); // recommend 1MiB starting internal buffer for 512KB blocks (double block size)
        Self {
            encode,
            sign_context: RansEncContext::new(SIGN_ALPH_SIZE),
            exponent_context: RansEncContext::new(EXP_ALPH_SIZE),
            mantissa_context: RansEncContext::new(MANT_ALPH_SIZE + 1),
            encoder,
        }
    }
    pub fn encode_values(&mut self ) -> (Vec<u8>, Vec<u8>, /*DEBUG CODE*/Vec<f32>) {
        let mut raw_symbols: Vec<u8> = vec![];

        //complete forward pass concurrently for the 3 parts.
        let (mut sign_vals, mut exp_vals, mut mant_vals, /*DEBUG CODE*/quantized) = self.componentize_forward_pass();


        println!("Beginning Backward Pass");
        let (mut r_sign, mut r_exp, mut r_mant) = (0, 0, 0);
        if let Some(idx) = self.sign_context.rescale_location.pop() {
            r_sign = idx;
        }
        if let Some(idx) = self.exponent_context.rescale_location.pop() {
            r_exp = idx;
        }
        if let Some(idx) = self.mantissa_context.rescale_location.pop() {
            r_mant = idx;
        }



        let mut sign_symbols : Vec<B64RansEncSymbol> = self.sign_context.snapshots.pop().unwrap_or_else(|| panic!("Failed to get sign context snapshot"));
        let mut exp_symbols : Vec<B64RansEncSymbol> = self.exponent_context.snapshots.pop().unwrap_or_else(|| panic!("Failed to get exp context snapshot"));
        let mut mant_symbols : Vec<B64RansEncSymbol> = self.mantissa_context.snapshots.pop().unwrap_or_else(|| panic!("Failed to get mant context snapshot"));

        let (mut in_range, mut out_range) = (0,0);

        for i in (0..self.encode.len()).rev() {
            if i == r_sign && i != 0 {
                // println!("Rescaling Sign Context Model");
                if let Some(idx) = self.sign_context.rescale_location.pop() {
                    r_sign = idx;
                }

                sign_symbols = self.sign_context.snapshots.pop().unwrap_or_else(|| panic!("Failed to get context snapshot"));
            }
            if i == r_exp && i != 0 {
                // println!("Rescaling Exponent Context Model");
                if let Some(idx) = self.exponent_context.rescale_location.pop() {
                    r_exp = idx;
                }
                exp_symbols = self.exponent_context.snapshots.pop().unwrap_or_else(|| panic!("Failed to get context snapshot"));
            }
            if i == r_mant && i != 0 {
                // println!("Rescaling Mantissa Context Model");
                if let Some(idx) = self.mantissa_context.rescale_location.pop() {
                    r_mant = idx;
                }
                mant_symbols = self.mantissa_context.snapshots.pop().unwrap_or_else(|| panic!("Failed to get context snapshot"));
            }



            if let Some(sign) = sign_vals.pop() {
                self.encoder.put_at(ENC_SIGN_CHANNEL,&sign_symbols[sign as usize]);
            }
            if let Some(exp) = exp_vals.pop() {
                if exp == 255 {
                    println!("Exp Symbol: {:?} at idx {}", exp, i);
                }
                self.encoder.put_at(ENC_EXPONENT_CHANNEL,&exp_symbols[exp as usize]);
            }
            if let Some(mant) = mant_vals.pop() {
                let mq_idx = RansEnc::quantize_idx(mant);
                match mq_idx == MANT_ALPH_SIZE as u32 {
                    true => {
                        self.encoder.put_at(ENC_MANTISSA_CHANNEL,&mant_symbols[MANT_ALPH_SIZE]);
                        raw_symbols.append(&mut mant.to_ne_bytes().to_vec());
                        out_range += 1;
                        // println!("Out of range mantissa: {}", RansEnc::quantize_idx(mant));
                    },
                    false => {
                        self.encoder.put_at(ENC_MANTISSA_CHANNEL,&mant_symbols[mq_idx as usize]);
                        in_range += 1;
                        // println!("In range mantissa: {}", RansEnc::quantize_idx(mant));
                    }
                }
            }
        }
        println!("Completed Backward Pass\n");
        println!("Mantissa in range {:?}, Mantissas out of range {:?}", in_range, out_range);

        self.encoder.flush_all();

        (self.encoder.data().to_owned(), raw_symbols, /*DEBUG CODE*/ quantized)

    }

    fn componentize_forward_pass(&mut self) -> (BitVec, Vec<u8>, Vec<u32>, /*DEBUG CODE*/ Vec<f32>) {
        println!("Beginning Forward Pass on contents length {:?}", self.encode.len());
        let mut signs : BitVec = BitVec::with_capacity(self.encode.len() as u64);
        let mut exponents : Vec<u8> = Vec::with_capacity(self.encode.len());
        let mut mantissa_bits = Vec::with_capacity(self.encode.len());

        //NOTE!: for debugging:
        let mut quantized: Vec<f32> = Vec::with_capacity(self.encode.len());
        //END! debug code

        self.sign_context.build_snapshot();
        self.exponent_context.build_snapshot();
        self.mantissa_context.build_snapshot();

        for (i,res)  in self.encode.iter().enumerate() {

            let bits = res.to_bits();

            let sign :bool = ((bits >> 31) & 1) != 0;
            let exponent = ((bits >> 23) & 0xFF) as u8;
            let mantissa = bits & 0x7F_FFFF;

            self.sign_context.increment_freq(sign as u32, i, ENC_SIGN_CHANNEL);
            self.exponent_context.increment_freq(exponent as u32, i, ENC_EXPONENT_CHANNEL);


            let mut mq_idx = RansEnc::quantize_idx(mantissa);
            if mq_idx < MANT_ALPH_SIZE as u32 {
                self.mantissa_context.increment_freq(mq_idx, i, ENC_MANTISSA_CHANNEL);
            }

            //NOTE! debug code:
            // let mut q_m = RansEnc::quantize_idx(mantissa);
            if mq_idx == MANT_ALPH_SIZE as u32 {mq_idx = mantissa} else {mq_idx = mq_idx * STEP as u32}
            let bits =
                ((sign as u32 & 0x1) << 31) |      // sign bit at bit 31
                    ((exponent as u32 & 0xFF) << 23) | // exponent in bits 23–30
                    (mq_idx & 0x7F_FFFF);
            let quantized_val = f32::from_bits(bits);
            quantized.push(quantized_val);
            //END! debug code

            signs.push(sign);
            exponents.push(exponent);
            mantissa_bits.push(mantissa) ;
        };

        println!("Completed Forward Pass\n");

        (signs, exponents, mantissa_bits, /*DEBUG CODE*/ quantized)
    }

    fn quantize_idx(m : u32) -> u32 {

        let rat : f32 = m as f32 / STEP;
        let mut idx = rat.round() as u32;

        let err = (m - (idx * STEP as u32)) as i32;
        if err.abs() >= MAX_ERR {
           idx = MANT_ALPH_SIZE as u32;
        }

        idx
    }
}

struct RansDecContext {
    context: Context,
    symbols: Vec<B64RansDecSymbol>,
    freq_to_symbol: Vec<usize>,
    alphabet_len: usize,
}


/**
 * Description RansDec implements the rANS decoding algorithm
 *
 * Decoding Loop:
 *  - keeps a adaptive context model and replaces its decoding histogram whenever the underlying
 *    context model total frequency equals 2^SCALE_BIT
 *  - returns an array of all encoded symbols
 */
impl RansDecContext {
    pub fn new(alphabet_len: usize) -> Self {
        let context = Context::new(alphabet_len);
        Self { context, symbols: vec![], freq_to_symbol: vec![], alphabet_len }
    }

    pub fn increment_freq(&mut self, symbol: usize) {
        self.context.increment_freq(symbol);
    }


    pub fn rebuild_histogram(&mut self, symbol: usize, channel: usize, idx: usize) {
        if self.context.rebuild_histogram() {
            // println!("rebuilding histogram idx {} for channel {}", idx, channel);
            self.build_inverse_freq_table();
            self.context.rescale_model();
        }
    }

    pub fn build_inverse_freq_table(&mut self) {
        self.symbols = Vec::with_capacity(self.alphabet_len);
        let mut cum_freqs = Vec::with_capacity(self.alphabet_len);
        let total_freq = 1 << SCALE_BIT;


        let mut cum_freq = 0;
        self.context.get_freq_array().iter().for_each(|x| {
            self.symbols.push(B64RansDecSymbol::new(
                cum_freq,
                *x as u32,
            ));
            cum_freqs.push(cum_freq);
            cum_freq += *x as u32;
        });

        self.freq_to_symbol = Vec::with_capacity(total_freq as usize);
        for i in 0..cum_freqs.len() - 1 {
            self.freq_to_symbol.resize(cum_freqs[i + 1] as usize, i);
        }
        self.freq_to_symbol.resize(total_freq as usize, cum_freqs.len() - 1);

    }

}

const DEC_SIGN_CHANNEl: usize = 2;
const DEC_EXPONENT_CHANNEL: usize = 1;
const DEC_MANTISSA_CHANNEL: usize = 0;
pub struct RansDec<'a> {
    sign_context: RansDecContext,
    exponent_context: RansDecContext,
    mantissa_context: RansDecContext,
    decoder: B64RansDecoderMulti<'a,3>,
    raw_bytes: Vec<u8>,
}

impl<'a> RansDec<'a> {
    pub fn new(code_data: &'a mut [u8], raw_bytes: Vec<u8>) -> Self {
        Self {
            sign_context: RansDecContext::new(SIGN_ALPH_SIZE),
            exponent_context: RansDecContext::new(EXP_ALPH_SIZE),
            mantissa_context: RansDecContext::new(MANT_ALPH_SIZE + 1),
            decoder: B64RansDecoderMulti::new(code_data),
            raw_bytes,
        }
    }
    pub fn decode_values(&mut self, length: usize) -> Vec<f32> {
        let mut res = Vec::with_capacity(length);
        self.sign_context.build_inverse_freq_table();
        self.exponent_context.build_inverse_freq_table();
        self.mantissa_context.build_inverse_freq_table();

        println!("\nBeginning Decoding");
        let (mut num_esc, mut num_code) = (0, 0);
        for i in 0..length {
            let sign_cum_freq = self.decoder.get_at(DEC_SIGN_CHANNEl, SCALE_BIT);
            let exp_cum_freq = self.decoder.get_at(DEC_EXPONENT_CHANNEL, SCALE_BIT);
            let mant_cum_freq = self.decoder.get_at(DEC_MANTISSA_CHANNEL, SCALE_BIT);
            // println!("cumulative freq {:?} {:?} {:?}", sign_cum_freq, exp_cum_freq, mant_cum_freq);

            let sign_symbol = self.sign_context.freq_to_symbol[sign_cum_freq as usize];
            let exp_symbol = self.exponent_context.freq_to_symbol[exp_cum_freq as usize];
            let mant_symbol = self.mantissa_context.freq_to_symbol[mant_cum_freq as usize];



            // println!("mant symbol {:?}", mant_symbol);



            self.sign_context.increment_freq(sign_symbol);
            self.exponent_context.increment_freq(exp_symbol);


            let mantissa: u32  = match mant_symbol < MANT_ALPH_SIZE  {
                false => {
                    let val = u32::from_ne_bytes(self.raw_bytes[self.raw_bytes.len() - 4..].try_into().unwrap());
                    self.raw_bytes.drain(self.raw_bytes.len() - 4..);
                    // println!("decoded out-of-range symbol: {:?}", val);
                    num_esc += 1;
                    val

                },
                true => {
                    self.mantissa_context.increment_freq(mant_symbol);
                    num_code += 1;
                    //convert from quantized index to raw value
                    // println!("Decoded quantized symbol: {:?}", mant_symbol);
                    mant_symbol as u32 * STEP as u32
                },
            };

            self.decoder.advance_step_at(DEC_SIGN_CHANNEl,&self.sign_context.symbols[sign_symbol], SCALE_BIT);
            self.decoder.advance_step_at(DEC_EXPONENT_CHANNEL,&self.exponent_context.symbols[exp_symbol], SCALE_BIT);
            self.decoder.advance_step_at(DEC_MANTISSA_CHANNEL,&self.mantissa_context.symbols[mant_symbol], SCALE_BIT);
            self.decoder.renorm_all();

            self.sign_context.rebuild_histogram(sign_symbol, DEC_SIGN_CHANNEl, i);
            self.exponent_context.rebuild_histogram(exp_symbol, DEC_EXPONENT_CHANNEL, i);
            self.mantissa_context.rebuild_histogram(mant_symbol, DEC_MANTISSA_CHANNEL, i);

            // println!("Decoded symbol: {:?} {:?} {:?}", sign_symbol as u32, exp_symbol, mantissa);

            let bits =
                    ((sign_symbol as u32 & 0x1) << 31) |
                    ((exp_symbol as u32 & 0xFF) << 23) |
                    (mantissa & 0x7F_FFFF);

            res.push(f32::from_bits(bits));
        }

        // println!("Decoded esc {} and coded {} ", num_esc, num_code);
        res
    }
}
