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
const SCALE_BIT: u32 = 14;
// const MAX_ERR : i32 = 1975;
const MAX_ERR : i32 = 1500;
const STEP : f32 = (1 << 11) as f32;
const MANT_ALPH_SIZE: usize = (((1 << 23) / STEP as usize) * 2) + 1;


struct Context {
    freq: Vec<u16>,
    total_freq: usize,
}

/**
 * Adaptive Frequency Context
 *
 * Maintains raw symbol-frequency counts for a single entropy-coding channel.
 * The encoder and decoder both use this structure as the underlying adaptive
 * model for their probability distributions.
 *
 * Responsibilities:
 *   - Store frequency counts for each symbol in the alphabet
 *   - Track total frequency and detect when rescaling is required
 *   - Rescale all frequencies by halving (clamping to 1) when total reaches 2^SCALE_BIT
 *
 * The rANS encoder/decoder layer builds cumulative or inverse tables from this
 * frequency model as needed.
 */

impl Context {
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

    /**
    * Returns `true` when the context model has accumulated enough total frequency
    * to trigger a histogram rebuild (i.e., total_freq == 2^SCALE_BIT).
    *
    * The caller is responsible for:
    *   - Building rANS symbol tables
    *   - Rescaling the model (halving all counts)
    *   - Snapshotting encoder tables if needed
    */
    pub fn rebuild_histogram(&self) -> bool {
       self.total_freq == (1 << SCALE_BIT)
    }


    /**
    * Rescales the model by halving all frequencies and clamping any zero values to 1.
    *
    * This prevents integer overflow in cumulative frequencies and keeps adaptation
    * responsive in long streams. The total frequency is recomputed after rescaling.
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
 * Adaptive rANS Encoding Context
 *
 * Holds:
 *   - An adaptive Context (raw frequencies)
 *   - A list of histogram snapshots taken during the forward pass
 *   - A list of indices where rescaling events occurred
 *   - A shift range to re-center exponent/mantissa deltas around zero
 *
 * The encoder uses this structure to:
 *   - Accumulate symbol frequencies in forward order (decoder order)
 *   - Take snapshots of cumulative frequencies whenever rescaling occurs
 *   - Restore those snapshots during the backward rANS encoding pass
 *
 * This ensures perfect synchronization between encoder and decoder state.
 */
impl RansEncContext {
    pub fn new(alphabet_len: usize, shift_range: i32) -> Self {
        let context = Context::new(alphabet_len);
        Self { context, snapshots: vec![], rescale_location: vec![] , shift_range}
    }

    pub fn shift_idx(&self, idx: i32) -> usize {
        (idx + self.shift_range) as usize
    }

    /**
    * Adds one occurrence of a symbol during the forward pass.
    *
    * When total_freq reaches 2^SCALE_BIT:
    *   - A frequency snapshot is captured for later backward-pass encoding
    *   - The underlying Context is rescaled
    *   - The current index is recorded so the backward pass knows when
    *     to restore the corresponding snapshot
    */

    pub fn increment_freq(&mut self, val: usize, idx: usize, channel: usize){
        self.context.increment_freq(val);
        if self.context.rebuild_histogram() {
            self.build_snapshot();
            self.context.rescale_model();
            self.rescale_location.push(idx);
        }
    }

    /**
    * Builds a cumulative-frequency snapshot for all symbols in the current context.
    *
    * The snapshot is expressed as a list of `B64RansEncSymbol` objects and captures:
    *   - Cumulative frequency at each symbol
    *   - Symbol frequency
    *   - Coding precision (SCALE_BIT)
    *
    * Snapshots are pushed onto a stack and later popped during the backward rANS pass.
    */

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

/**
 * rANS Encoder
 *
 * Encodes a stream of `f32` values by:
 *   1. Componentizing each float into (sign, exponent, mantissa)
 *   2. Quantizing mantissas into indices with bounded error
 *   3. Escaping large-error mantissas into a secondary raw-byte buffer
 *   4. Delta-encoding exponent and mantissa-index streams
 *   5. Running a forward pass to build adaptive frequency snapshots
 *   6. Running a backward rANS pass to emit compressed bytes
 *
 */

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

    /**
    * Performs full rANS encoding over the input float stream.
    *
    * Produces:
    *   - The rANS-coded bytes for the componentized stream
    *   - A buffer of raw escaped mantissas (only when quantization error > MAX_ERR)
    *   - A debug vector of quantized reconstructed floats
    *
    * Internally:
    *   - Executes a forward pass to build frequency snapshots
    *   - Executes a backward pass applying rANS encoding using correct snapshots
    *   - Writes escape symbols and raw mantissa bytes as needed
    */

    pub fn encode_values(&mut self ) -> (Vec<u8>, Vec<u8>, /*DEBUG CODE*/Vec<f32>) {
        let mut raw_symbols: Vec<u8> = vec![];

        //complete forward pass concurrently for the 3 parts.
        let (mut components, /*DEBUG CODE*/quantized) = self.forward_pass();


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


            if let Some(comp) = components.pop() {
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
        self.encoder.flush_all();

        (self.encoder.data().to_owned(), raw_symbols, /*DEBUG CODE*/ quantized)

    }

    /**
    * Forward Pass (Encoder Simulation of Decoder Order)
    *
    * Steps:
    *   1. Build initial snapshots for all three channels
    *   2. Componentize floats into (sign, exponent, quantized mantissa index)
    *   3. Delta-encode exponent and mantissa-index streams
    *   4. Feed symbols in forward order into the adaptive contexts
    *   5. Capture snapshots whenever rescaling occurs
    *
    * This prepares the encoder for correct backward rANS encoding by ensuring that
    * the same histogram transitions will occur in the decoder.
    */

    fn forward_pass(&mut self) -> (Vec<(bool, i16, i32, Option<u32>)>, /*DEBUG CODE*/ Vec<f32>) {
        self.sign_context.build_snapshot();
        self.exponent_context.build_snapshot();
        self.mantissa_context.build_snapshot();

        //1. Componentize float and compute quantized mantissa idx
        let mut components = RansEnc::componentize_and_quantize(self.encode);

        //NOTE! debug code:
        let quantized = RansDec::rebuild_floats(&components);

        //2. Delta encode exponent and mantissa idx
        RansEnc::delta_encode(&mut components);


        //3. emulate decoder forward pass and caching
        for (i,comp)  in components.iter().enumerate() {

            self.sign_context.increment_freq(comp.0 as usize, i, ENC_SIGN_CHANNEL);
            self.exponent_context.increment_freq(self.exponent_context.shift_idx(comp.1 as i32), i, ENC_EXPONENT_CHANNEL);

            if comp.3 == None {
                self.mantissa_context.increment_freq(self.mantissa_context.shift_idx(comp.2), i, ENC_MANTISSA_CHANNEL);
            }
        };


        (components, /*DEBUG CODE*/ quantized)

    }

    /**
    * Converts each `f32` into (sign, exponent, quantized_mantissa_index, raw_escape_opt).
    *
    * Mantissa is quantized with:
    *      qm_idx = round(mantissa / STEP)
    *
    * If the quantized mantissa deviates from the original by ≥ MAX_ERR, the mantissa is
    * marked as out-of-range and stored in `components.3` to be emitted as raw bytes.
    *
    * Returned structure:
    *   (sign_bit, exponent, quantized_index, optional_raw_mantissa)
    */

    fn componentize_and_quantize(vals: &Vec<f32>) -> Vec<(bool, i16, i32, Option<u32>)> {
        let mut res : Vec<(bool, i16, i32, Option<u32>)> = Vec::with_capacity(vals.len());
        let mut i = 0;
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

        res
    }

    /**
    * First-order delta encoding over exponent and quantized mantissa index.
    *
    * For each i > 0:
    *   exp[i]  ← exp[i]  − exp[i−1]
    *   idx[i]  ← idx[i]  − idx[i−1]
    *
    * Sign bits and raw-escaped mantissas are not delta-coded.
    *
    * This reduces symbol variance, improving rANS compression efficiency.
    */

    fn delta_encode(components: &mut Vec<(bool, i16, i32, Option<u32>)>) {
        if components.len() < 2 { return; }
        for i in (1..components.len()).rev() {
            //delta code exponent
            components[i].1 = components[i].1 - components[i-1].1;
            //delta code mantissa idx
            components[i].2 = components[i].2 - components[i-1].2;

        }
    }

    // fn second_order_delta_encode(comps: &mut Vec<(bool, i16, i32, Option<u32>)>) {
    //     if comps.len() < 3 { return; }
    //     for i in (2..comps.len()).rev() {
    //         comps[i].1 = comps[i].1 - (comps[i-1].1 + (comps[i-1].1 - comps[i-2].1));
    //         comps[i].2 = comps[i].2 - (comps[i-1].2 + (comps[i-1].2 - comps[i-2].2));
    //     }
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
 * Adaptive rANS Decoding Context
 *
 * Mirrors `RansEncContext` but builds inverse-frequency tables rather than
 * forward-pass snapshots. Each context:
 *
 *   - Tracks adaptive symbol frequencies
 *   - Builds cumulative → symbol lookup tables for decoding
 *   - Rescales frequencies when total reaches 2^SCALE_BIT
 *
 * The decoder updates its context symbol-by-symbol to match the encoder's
 * forward pass exactly.
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
            self.build_inverse_freq_table();
            self.context.rescale_model();
        }
    }

    /**
    * Builds tables used by the rANS decoder:
    *   - `symbols`: cumulative-frequency table of `B64RansDecSymbol`
    *   - `freq_to_symbol`: reverse lookup table mapping ranges of cumulative
    *       frequencies back to symbol indices
    *
    * These tables allow the decoder to identify symbols from the rANS state.
    */

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

/**
 * rANS Decoder
 *
 * Reconstructs the original sequence of floats by:
 *   1. Running adaptive rANS decoding over sign, exponent, and mantissa-index channels
 *   2. Reading raw escaped mantissas when escape symbols are encountered
 *   3. Undoing delta prediction over exponent and index channels
 *   4. Rebuilding IEEE-754 floats from decoded components
 *
 * The decoder maintains adaptive contexts that evolve in the same order as the encoder’s
 * forward pass, ensuring both stay perfectly synchronized.
 */
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

    /**
    * Decodes `length` components using multi-channel adaptive rANS.
    *
    * For each symbol position:
    *   - Queries cumulative frequencies from each rANS channel
    *   - Maps cumulative frequency → symbol via inverse frequency tables
    *   - Updates adaptive frequency models
    *   - Advances the rANS decoder state on all three channels
    *
    * After all components are decoded:
    *   - Runs delta decoding to restore exponents and quantized mantissa indices
    *   - Rebuilds floats using quantized or raw mantissas
    */

    pub fn decode_values(&mut self, length: usize) -> Vec<f32> {
        let mut components : Vec<(bool, i16, i32, Option<u32>)> = Vec::with_capacity(length);
        self.sign_context.build_inverse_freq_table();
        self.exponent_context.build_inverse_freq_table();
        self.mantissa_context.build_inverse_freq_table();

        for i in 0..length {
            let mut component : (bool, i16, i32, Option<u32>);

            let sign_cum_freq = self.decoder.get_at(DEC_SIGN_CHANNEl, SCALE_BIT);
            let exp_cum_freq = self.decoder.get_at(DEC_EXPONENT_CHANNEL, SCALE_BIT);
            let mant_cum_freq = self.decoder.get_at(DEC_MANTISSA_CHANNEL, SCALE_BIT);

            let sign_symbol = self.sign_context.freq_to_symbol[sign_cum_freq as usize];
            let mut exp_symbol = self.exponent_context.freq_to_symbol[exp_cum_freq as usize];
            let mut mant_symbol = self.mantissa_context.freq_to_symbol[mant_cum_freq as usize];


            self.sign_context.increment_freq(sign_symbol);
            self.exponent_context.increment_freq(exp_symbol);

            component = ((sign_symbol & 1) != 0, self.exponent_context.shift_idx(exp_symbol) as i16, self.mantissa_context.shift_idx(mant_symbol), None );

            if mant_symbol < MANT_ALPH_SIZE {
                self.mantissa_context.increment_freq(mant_symbol);
            } else {
                let val = u32::from_ne_bytes(self.raw_bytes[self.raw_bytes.len() - 4..].try_into().unwrap());
                self.raw_bytes.drain(self.raw_bytes.len() - 4..);
                component.2 = (val as f32 / STEP).round() as i32;
                component.3 = Some(val);
            }

            components.push(component);


            self.decoder.advance_step_at(DEC_SIGN_CHANNEl,&self.sign_context.symbols[sign_symbol], SCALE_BIT);
            self.decoder.advance_step_at(DEC_EXPONENT_CHANNEL,&self.exponent_context.symbols[exp_symbol], SCALE_BIT);
            self.decoder.advance_step_at(DEC_MANTISSA_CHANNEL,&self.mantissa_context.symbols[mant_symbol], SCALE_BIT);
            self.decoder.renorm_all();

            self.sign_context.rebuild_histogram(sign_symbol, DEC_SIGN_CHANNEl, i);
            self.exponent_context.rebuild_histogram(exp_symbol, DEC_EXPONENT_CHANNEL, i);
            self.mantissa_context.rebuild_histogram(mant_symbol, DEC_MANTISSA_CHANNEL, i);

        }

        RansDec::delta_decode(&mut components);

        RansDec::rebuild_floats(&components)

    }

    /**
    * Reverses the first-order delta encoding applied during encoding.
    *
    * For each i > 0:
    *   exp[i]  ← exp[i]  + exp[i−1]
    *   idx[i]  ← idx[i]  + idx[i−1], unless the symbol was an escape
    *
    * This restores the original exponent and quantized mantissa index streams.
    */

    fn delta_decode(comp: &mut Vec<(bool, i16, i32, Option<u32>)>) {
        if comp.len() < 2 { return; }
        for i in 1..comp.len() {
            comp[i].1 = comp[i].1 + comp[i-1].1;
            if comp[i].3 == None { comp[i].2 = comp[i].2 + comp[i-1].2; }
        }
    }

    /**
    * Reconstructs IEEE-754 floats from:
    *   - sign bit
    *   - recovered exponent
    *   - mantissa (either quantized or raw)
    *
    * Performs:
    *   bits = (sign << 31) | (exponent << 23) | (mantissa & 0x7FFFFF)
    *
    * Returns a vector of fully reconstructed `f32` values.
    */

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
        res
    }

    // fn second_order_delta_decode(comp: &mut Vec<(bool, i16, i32, Option<u32>)>) {
    //     if comp.len() < 3 { return; }
    //     for i in 2..comp.len() {
    //         comp[i].1 = comp[i].1 + (comp[i-1].1 + (comp[i-1].1 - comp[i-2].1));
    //         if comp[i].3 == None { comp[i].2 = comp[i].2 + (comp[i-1].2 + (comp[i-1].2 - comp[i-2].2)); }
    //     }
    // }
}
