/**
 * File Name: rans_coding
 * Description: This file implements a block-adaptive rANS coder and decoder
 * Date Created: 11/05/2025
 * Date Last Modified: 11/05/2025
 */

use rans::b64_encoder::{B64RansEncSymbol, B64RansEncoder, B64RansEncoderMulti};
use rans::{RansEncSymbol, RansEncoder, RansEncoderMulti, RansDecoder, RansDecSymbol};
use rans::b64_decoder::{B64RansDecoder, B64RansDecSymbol};
use bv::BitVec;

const SIGN_ALPH_SIZE : usize = 2;
const EXP_ALPH_SIZE : usize = 1 << 8;
const MANT_ALPH_SIZE: u32 = 257;
const SCALE_BIT: u32 = 14;
const MAX_ERR : i32 = 15_000;
const SIGN_CHANNEL: usize = 0;
const EXPONENT_CHANNEL: usize = 1;
const MANTISSA_CHANNEL: usize = 2;

struct Context {
    alphabet_len: usize,
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
        Self{ alphabet_len,
            freq: vec![1; alphabet_len  + 1],
            total_freq: alphabet_len + 1,
            }
    }

    //TODO: consider whether we should "adapt" aka increment frequency of the escape symbol
    //may not matter much. May be better determined with testing
    //currently we do not increment the escape character
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
    alphabet_len: usize,
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
        Self { context, snapshots: vec![], rescale_location: vec![], alphabet_len}
    }

    //might want to add this into component breakdown
    pub fn increment_freq(&mut self, val: u32, idx: usize){

        if self.context.rebuild_histogram() {
            self.build_snapshot();
            self.context.rescale_model();
            self.rescale_location.push(idx);
        }

        if idx <= self.alphabet_len {
            self.context.increment_freq(val as usize);
        }

        // match self.context.shift_range(val) {
        //     (true, idx) => {
        //         self.context.increment_freq(idx);
        //     },
        //     //this skips incrementing frequency if out of range
        //     _ => { },
        // }
    }

    pub fn build_snapshot(&mut self) {
        // println!("build snapsht");
        let res : Vec<B64RansEncSymbol> = self.context.get_freq_array().iter()
            .scan(0u32, |acc, &x|{
                let val = *acc;
                *acc += x as u32;
                // println!("new FREQ: {}", x);
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
            //TODO determine correct alphabet sizes
            encode,
            sign_context: RansEncContext::new(SIGN_ALPH_SIZE),
            exponent_context: RansEncContext::new(EXP_ALPH_SIZE),
            mantissa_context: RansEncContext::new(MANT_ALPH_SIZE as usize),
            encoder,
        }
    }
    pub fn encode_values(&mut self, values: &Vec<i32> ) -> (Vec<u8>, Vec<u8>) {
        let mut raw_symbols: Vec<u8> = vec![];

        //complete forward pass concurrently for the 3 parts.
        let (mut sign_vals, mut exp_vals, mut mant_vals) = self.componentize_forward_pass();


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

        let mut sign_symbols : Vec<B64RansEncSymbol> = self.sign_context.snapshots.pop().unwrap_or_else(|| panic!("Failed to get context snapshot"));
        let mut exp_symbols : Vec<B64RansEncSymbol> = self.sign_context.snapshots.pop().unwrap_or_else(|| panic!("Failed to get context snapshot"));
        let mut mant_symbols : Vec<B64RansEncSymbol> = self.sign_context.snapshots.pop().unwrap_or_else(|| panic!("Failed to get context snapshot"));

        for i in 0..values.len() {

            if let Some(sign) = sign_vals.pop() {
                self.encoder.put_at(SIGN_CHANNEL,&sign_symbols[sign as usize]);
            }
            if let Some(exp) = exp_vals.pop() {
               self.encoder.put_at(EXPONENT_CHANNEL,&exp_symbols[exp as usize]);
            }
            if let Some(mant) = mant_vals.pop() {
               match mant == MANT_ALPH_SIZE {
                   true => {
                       self.encoder.put_at(MANTISSA_CHANNEL,&mant_symbols[MANT_ALPH_SIZE as usize]);
                       raw_symbols.append(&mut mant.to_ne_bytes().to_vec());
                   },
                   false => {
                       self.encoder.put_at(MANTISSA_CHANNEL,&mant_symbols[mant as usize]);
                   }
               }
            }

            if i == r_sign && i != 0 {
                if let Some(idx) = self.sign_context.rescale_location.pop() {
                    r_sign = idx;
                }
                sign_symbols = self.sign_context.snapshots.pop().unwrap_or_else(|| panic!("Failed to get context snapshot"));
            }
            if i == r_exp && i != 0 {
                if let Some(idx) = self.exponent_context.rescale_location.pop() {
                    r_exp = idx;
                }
                exp_symbols = self.exponent_context.snapshots.pop().unwrap_or_else(|| panic!("Failed to get context snapshot"));
            }
            if i == r_mant && i != 0 {
                if let Some(idx) = self.mantissa_context.rescale_location.pop() {
                    r_mant = idx;
                }
                mant_symbols = self.mantissa_context.snapshots.pop().unwrap_or_else(|| panic!("Failed to get context snapshot"));
            }
        }
        println!("Completed Backward Pass\n");

        self.encoder.flush_all();

        (self.encoder.data().to_owned(), raw_symbols)

    }

    fn componentize_forward_pass(&mut self) -> (BitVec, Vec<u8>, Vec<u32>) {
        println!("Beginning Forward Pass");

        let mut signs : BitVec = BitVec::with_capacity(self.encode.len() as u64);
        let mut exponents : Vec<u8> = Vec::with_capacity(self.encode.len());
        let mut mantissa_bits = Vec::with_capacity(self.encode.len());

        self.sign_context.build_snapshot();
        self.exponent_context.build_snapshot();
        self.mantissa_context.build_snapshot();

        //first value not delta coded
        if let Some(val) = self.encode.first() {
            let bits = val.to_bits();

            let sign = ((bits >> 31) & 1) != 0;
            let exponent = ((bits >> 23) & 0xFF) as u8;
            let mantissa = bits & 0x7F_FFFF;

            //right will definetly need to figure out
            self.sign_context.increment_freq(sign as u32, 0);
            self.exponent_context.increment_freq(exponent as u32, 0);
            self.mantissa_context.increment_freq(mantissa, 0);


            signs.push(sign);
            exponents.push(exponent);
            mantissa_bits.push(mantissa);
        }

        //delta code exponent and mantissa of remaining values
        if self.encode.len() > 1 {
            for i  in 1..self.encode.len() {
                let residual  = (self.encode[i] - self.encode[i - 1]);

                // print!("{:?},", residual);

                let bits = residual.to_bits();

                let sign :bool = ((bits >> 31) & 1) != 0;
                let exponent = ((bits >> 23) & 0xFF) as u8;
                let mantissa = bits & 0x7F_FFFF;

                self.sign_context.increment_freq(sign as u32, i);
                self.exponent_context.increment_freq(exponent as u32, i);
                self.mantissa_context.increment_freq(RansEnc::quantize_idx(mantissa), i);

                signs.push(sign);
                exponents.push(exponent);
                mantissa_bits.push(mantissa) ;
            };
        }

        println!("Completed Forward Pass\n");

        (signs, exponents, mantissa_bits)
    }

    fn quantize_idx(m : u32) -> u32 {
        let step = 1 << 15;

        let rat : f32 = m as f32 / step as f32;
        let mut idx = rat.round() as u32;

        let err = (m - (idx * step)) as i32;
        if err.abs() <= MAX_ERR {
           idx = MANT_ALPH_SIZE;
        }

        idx
    }
}

pub struct RansDecContext<'a> {
    context: Context,
    // decoder: B64RansDecoder<'a>,
    raw_bytes: Vec<u8>,
    symbols: Vec<B64RansDecSymbol>,
    freq_to_symbol: Vec<usize>,
}


/**
 * Description RansDec implements the rANS decoding algorithm
 *
 * Decoding Loop:
 *  - keeps a adaptive context model and replaces its decoding histogram whenever the underlying
 *    context model total frequency equals 2^SCALE_BIT
 *  - returns an array of all encoded symbols
 */
impl<'a> RansDecContext<'a> {
    pub fn new(code_data: &'a mut [u8], raw_bytes: Vec<u8>) -> Self {
        let context = Context::new(NUM_SYMBOLS);
        let decoder = B64RansDecoder::new(code_data);
        Self { context, decoder, raw_bytes, symbols: vec![], freq_to_symbol: vec![] }
    }

    pub fn decode_values(&mut self, length: usize) -> Vec<i32> {
        let mut res = Vec::with_capacity(length);
        self.build_inverse_freq_table();

        println!("\nBeginning Decoding");
        for _ in 0..length {
            let cum_freq = self.decoder.get(SCALE_BIT);

            let symbol = self.freq_to_symbol[cum_freq as usize];

            //need to verify logic
            let value: i32  = match symbol == NUM_SYMBOLS {
                true => {
                    assert!(!self.raw_bytes.len() >= 4);
                    let val = i32::from_ne_bytes(self.raw_bytes[self.raw_bytes.len() - 4..].try_into().unwrap());
                    self.raw_bytes.drain(self.raw_bytes.len() - 4..);
                    println!("decoded out-of-range symbol: {:?}", val);
                    val

                },
                false => {
                    self.context.increment_freq(symbol);
                    symbol as i32 - SHIFT_RANGE
                },
            };

            res.push(value);

            self.decoder.advance(&self.symbols[symbol], SCALE_BIT);


            if self.context.rebuild_histogram() {
                // println!("rebuilding freq table");
                self.build_inverse_freq_table();
                self.context.rescale_model();
            }
        }

        res
    }

    pub fn build_inverse_freq_table(&mut self) {
        self.symbols = Vec::with_capacity(NUM_SYMBOLS);
        let mut cum_freqs = Vec::with_capacity(NUM_SYMBOLS);
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
