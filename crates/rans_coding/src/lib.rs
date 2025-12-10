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
const EXP_ALPH_SIZE : usize = (1 << 8) * 2;
const SCALE_BIT: u32 = 16;
// const MAX_ERR : i32 = 15_000;
const MAX_ERR : i32 = 3800;
const STEP : f32 = (1 << 13) as f32;
const MANT_ALPH_SIZE: usize = (((1 << 23) / STEP as usize) * 2) + 1;


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
    shift_range: i32,
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
    pub fn new(alphabet_len: usize, shift_range: i32) -> Self {
        let context = Context::new(alphabet_len);
        Self { context, snapshots: vec![], rescale_location: vec![] , shift_range}
    }

    pub fn shift_idx(&self, idx: i32) -> usize {
        (idx + self.shift_range) as usize
    }

    //might want to add this into component breakdown
    pub fn increment_freq(&mut self, val: usize, idx: usize, channel: usize){
        self.context.increment_freq(val);
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
            sign_context: RansEncContext::new(SIGN_ALPH_SIZE, 0),
            exponent_context: RansEncContext::new(EXP_ALPH_SIZE, (EXP_ALPH_SIZE / 2) as i32),
            mantissa_context: RansEncContext::new(MANT_ALPH_SIZE + 1, (MANT_ALPH_SIZE / 2) as i32 ),
            encoder,
        }
    }
    pub fn encode_values(&mut self ) -> (Vec<u8>, Vec<u8>, /*DEBUG CODE*/Vec<f32>) {
        let mut raw_symbols: Vec<u8> = vec![];

        //complete forward pass concurrently for the 3 parts.
        let (mut components, /*DEBUG CODE*/quantized) = self.forward_pass();


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


            if let Some(comp) = components.pop() {
                // println!("encoding comp {:?}", comp);
                //encode sign
                self.encoder.put_at(ENC_SIGN_CHANNEL,&sign_symbols[comp.0 as usize]);

                //encode exponent
                self.encoder.put_at(ENC_EXPONENT_CHANNEL,&exp_symbols[self.exponent_context.shift_idx(comp.1 as i32)]);

                //encode mant idx
                if comp.3 != None {
                    self.encoder.put_at(ENC_MANTISSA_CHANNEL,&mant_symbols[MANT_ALPH_SIZE]);
                    raw_symbols.append(&mut comp.3.unwrap().to_ne_bytes().to_vec());
                    out_range += 1;
                } else {
                    self.encoder.put_at(ENC_MANTISSA_CHANNEL,&mant_symbols[self.mantissa_context.shift_idx(comp.2)]);
                    in_range += 1;
                }
            }


        }
        println!("Completed Backward Pass");
        println!("Mantissa in range {:?}, Mantissas out of range {:?}\n", in_range, out_range);

        self.encoder.flush_all();

        (self.encoder.data().to_owned(), raw_symbols, /*DEBUG CODE*/ quantized)

    }

    fn forward_pass(&mut self) -> (Vec<(bool, i16, i32, Option<u32>)>, /*DEBUG CODE*/ Vec<f32>) {
        println!("Beginning Forward Pass on contents length {:?}", self.encode.len());

        self.sign_context.build_snapshot();
        self.exponent_context.build_snapshot();
        self.mantissa_context.build_snapshot();

        //1. Componentize float and compute quantized mantissa idx
        let mut components = RansEnc::componentize_and_quantize(self.encode);

        //NOTE! debug code:
        let quantized = RansDec::rebuild_floats(&components);

        //2. Delta encode exponent and mantissa idx
        self.delta_encode(&mut components);


        //3. emulate decoder forward pass and caching
        for (i,comp)  in components.iter().enumerate() {

            self.sign_context.increment_freq(comp.0 as usize, i, ENC_SIGN_CHANNEL);
            self.exponent_context.increment_freq(self.exponent_context.shift_idx(comp.1 as i32), i, ENC_EXPONENT_CHANNEL);

            if comp.3 == None {
                self.mantissa_context.increment_freq(self.mantissa_context.shift_idx(comp.2), i, ENC_MANTISSA_CHANNEL);
            }
        };

        println!("Completed Forward Pass");

        (components, /*DEBUG CODE*/ quantized)

    }

    fn componentize_and_quantize(vals: &Vec<f32>) -> Vec<(bool, i16, i32, Option<u32>)> {
        let mut res : Vec<(bool, i16, i32, Option<u32>)> = Vec::with_capacity(vals.len());
        let mut i = 0;
        println!("Beginning Componentization and Quantize {}", vals.len());
        for val in vals {
            let mut components : (bool, i16, i32, Option<u32>);

            let bits = val.to_bits();

            //extract sign, exponent, mantissa
            let sign : bool = ((bits >> 31) & 1) != 0;
            let exp = ((bits >> 23) & 0xFF) as i16;
            let mant = bits & 0x7F_FFFF;

            //compute nearest quantization step
            //NOTE: I will delta code my quantization idx NOT the mantissa.
            let qm_idx = (mant as f32 / STEP).round() as i32;

            components = (sign, exp, qm_idx, None);

            //check if the quantized mantissa is in range
            if (mant as i32 - (qm_idx* STEP as i32)).abs() >= MAX_ERR {
                // components.2 = MANT_ALPH_SIZE as i32 + 1;
                components.3 = Some(mant);
                i += 1;
            }
            res.push(components);
        };

        println!("Completed Componentization and Quantize\nMantissa out of range: {}", i);

        res
    }

    fn delta_encode(&self, components: &mut Vec<(bool, i16, i32, Option<u32>)>) {
        println!("Beginning Delta Encoding {}", components.len());
        if components.len() < 2 { return; }
        for i in (1..components.len()).rev() {
            //delta code exponent
            components[i].1 = components[i].1 - components[i-1].1;
            //delta code mantissa idx
            components[i].2 = components[i].2 - components[i-1].2;

        }
        println!("Completed Delta Encoding {:?}", &components[0..10]);
    }

    //Rewrite if use
    // fn second_order_delta_encode(v: &mut [u32]) {
    //     if v.len() < 3 { return; }
    //     for i in (2..v.len()).rev() {
    //         v[i] = v[i] - (v[i-1] + (v[i-1] - v[i-2]));
    //     }
    // }


    // fn quantize_idx(m : u32) -> u32 {
    //
    //     let mut idx = m / STEP as u32;
    //     if m % STEP as u32 != 0 {
    //         idx = MANT_ALPH_SIZE as u32;
    //     }
    //
    //     idx
    // }
}

struct RansDecContext {
    context: Context,
    symbols: Vec<B64RansDecSymbol>,
    freq_to_symbol: Vec<usize>,
    alphabet_len: usize,
    shift_range: i32,
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
    pub fn new(alphabet_len: usize, shift_range: i32) -> Self {
        let context = Context::new(alphabet_len);
        Self { context, symbols: vec![], freq_to_symbol: vec![], alphabet_len, shift_range }
    }

    pub fn shift_idx(&self, symbol: usize) -> i32 {
        symbol as i32 - self.shift_range
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
            sign_context: RansDecContext::new(SIGN_ALPH_SIZE, 0),
            exponent_context: RansDecContext::new(EXP_ALPH_SIZE, (EXP_ALPH_SIZE / 2) as i32),
            mantissa_context: RansDecContext::new(MANT_ALPH_SIZE + 1, (MANT_ALPH_SIZE / 2) as i32),
            decoder: B64RansDecoderMulti::new(code_data),
            raw_bytes,
        }
    }
    pub fn decode_values(&mut self, length: usize) -> Vec<f32> {
        let mut components : Vec<(bool, i16, i32, Option<u32>)> = Vec::with_capacity(length);
        self.sign_context.build_inverse_freq_table();
        self.exponent_context.build_inverse_freq_table();
        self.mantissa_context.build_inverse_freq_table();

        println!("\nBeginning Decoding");
        let (mut num_esc, mut num_code) = (0, 0);
        for i in 0..length {
            let mut component : (bool, i16, i32, Option<u32>);

            let sign_cum_freq = self.decoder.get_at(DEC_SIGN_CHANNEl, SCALE_BIT);
            let exp_cum_freq = self.decoder.get_at(DEC_EXPONENT_CHANNEL, SCALE_BIT);
            let mant_cum_freq = self.decoder.get_at(DEC_MANTISSA_CHANNEL, SCALE_BIT);
            // println!("cumulative freq {:?} {:?} {:?}", sign_cum_freq, exp_cum_freq, mant_cum_freq);

            let sign_symbol = self.sign_context.freq_to_symbol[sign_cum_freq as usize];
            let mut exp_symbol = self.exponent_context.freq_to_symbol[exp_cum_freq as usize];
            let mut mant_symbol = self.mantissa_context.freq_to_symbol[mant_cum_freq as usize];



            // println!("mant symbol {:?}", mant_symbol);

            self.sign_context.increment_freq(sign_symbol);
            self.exponent_context.increment_freq(exp_symbol);

            component = ((sign_symbol & 1) != 0, self.exponent_context.shift_idx(exp_symbol) as i16, self.mantissa_context.shift_idx(mant_symbol), None );

            if mant_symbol < MANT_ALPH_SIZE {
                self.mantissa_context.increment_freq(mant_symbol);
            } else {
                // println!("Decoded out of range symbol: {:?}", mant_symbol);
                let val = u32::from_ne_bytes(self.raw_bytes[self.raw_bytes.len() - 4..].try_into().unwrap());
                self.raw_bytes.drain(self.raw_bytes.len() - 4..);
                component.2 = (val as f32 / STEP).round() as i32;
                component.3 = Some(val);
                // println!("Decoded component: {:?}", component);
            }

            components.push(component);


            // let mut mantissa: u32  = match mant_symbol < MANT_ALPH_SIZE  {
            //     false => {
            //         let val = u32::from_ne_bytes(self.raw_bytes[self.raw_bytes.len() - 4..].try_into().unwrap());
            //         self.raw_bytes.drain(self.raw_bytes.len() - 4..);
            //         // println!("decoded out-of-range symbol: {:?}", val);
            //         num_esc += 1;
            //         val
            //
            //     },
            //     true => {
            //         self.mantissa_context.increment_freq(mant_symbol);
            //         num_code += 1;
            //         //convert from quantized index to raw value
            //         // println!("Decoded quantized symbol: {:?}", mant_symbol);
            //
            //
            //         mant_symbol as u32 * STEP as u32;
            //
            //
            //
            //     },
            // };

            self.decoder.advance_step_at(DEC_SIGN_CHANNEl,&self.sign_context.symbols[sign_symbol], SCALE_BIT);
            self.decoder.advance_step_at(DEC_EXPONENT_CHANNEL,&self.exponent_context.symbols[exp_symbol], SCALE_BIT);
            self.decoder.advance_step_at(DEC_MANTISSA_CHANNEL,&self.mantissa_context.symbols[mant_symbol], SCALE_BIT);
            self.decoder.renorm_all();

            self.sign_context.rebuild_histogram(sign_symbol, DEC_SIGN_CHANNEl, i);
            self.exponent_context.rebuild_histogram(exp_symbol, DEC_EXPONENT_CHANNEL, i);
            self.mantissa_context.rebuild_histogram(mant_symbol, DEC_MANTISSA_CHANNEL, i);

            // println!("Decoded symbol: {:?} {:?} {:?}", sign_symbol as u32, exp_symbol, mantissa);




        }

        RansDec::delta_decode(&mut components);

        // println!("Decoded esc {} and coded {} ", num_esc, num_code);
        RansDec::rebuild_floats(&components)

    }

    fn delta_decode(comp: &mut Vec<(bool, i16, i32, Option<u32>)>) {
        if comp.len() < 2 { return; }
        for i in 1..comp.len() {
            comp[i].1 = comp[i].1 + comp[i-1].1;
            if comp[i].3 == None { comp[i].2 = comp[i].2 + comp[i-1].2; }
            // comp[i].2 = comp[i].2 + comp[i-1].2;
        }
        println!("Decoded Delta Components {:?}", &comp[0..10]);
    }

    fn rebuild_floats(components: &Vec<(bool, i16, i32, Option<u32>)>) -> Vec<f32> {
        let mut res : Vec<f32> = Vec::with_capacity(components.len());
        for comp in components {
            let mant : i32 = match comp.3 != None {
                true => {
                    comp.3.unwrap() as i32
                },
                false => comp.2 * STEP as i32};
            let bits =
                    ((comp.0 as u32 & 0x1) << 31) |
                    ((comp.1 as u32 & 0xFF) << 23) |
                    (mant as u32 & 0x7F_FFFF);
            res.push(f32::from_bits(bits));
        }
        println!("Decoded floats {:?}", res.len());
        res
    }

    fn second_order_delta_decode(v: &mut [f32]) {
        if v.len() < 3 { return; }
        for i in 2..v.len() {
            v[i] = v[i] + (v[i-1] + (v[i-1] - v[i-2]));
        }
    }
}
