/**
 * File Name: rans_coding
 * Description: This file implements a block-adaptive rANS coder and decoder
 * Date Created: 11/05/2025
 * Date Last Modified: 11/05/2025
 */

use rans::b64_encoder::{B64RansEncSymbol, B64RansEncoderMulti, B64RansEncoder};
use rans::{RansEncSymbol, RansEncoderMulti, RansDecSymbol, RansDecoderMulti, RansEncoder, RansDecoder};
use rans::b64_decoder::{B64RansDecSymbol, B64RansDecoderMulti, B64RansDecoder};

//im messing with stuff
use std::collections::HashMap;

const SIGN_ALPH_SIZE : usize = 2;
const EXP_ALPH_SIZE : usize = (1 << 8) * 2;


struct Context {
    freq: Vec<u16>,
    total_freq: usize,
    scale_bit: u32,
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
    pub fn new(alphabet_len: usize, scale_bit: u32) -> Self {
        Self{
            freq: vec![1; alphabet_len],
            total_freq: alphabet_len,
            scale_bit,
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
       self.total_freq == (1 << self.scale_bit)
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
    scale_bit: u32
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
    pub fn new(alphabet_len: usize, shift_range: i32, scale_bit: u32) -> Self {
        let context = Context::new(alphabet_len, scale_bit);
        Self { context, snapshots: vec![], rescale_location: vec![] , shift_range, scale_bit}
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

    pub fn increment_freq(&mut self, val: usize, idx: usize){
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
                self.scale_bit
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
    step: f32,
    mant_alph_size: usize,
    max_err: i32,
    sign_context: RansEncContext,
    exponent_context: RansEncContext,
    mantissa_context: RansEncContext,
    encoder: B64RansEncoderMulti<3>,

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

    //yup messing with stuff
    pub fn debug_stats(&self) {
        let mut comps = self.componentize_and_quantize(self.encode);
        let n = comps.len();
        if n == 0 { return; }

        // pre-delta copies (optional but helpful)
        // let exp0: Vec<i16> = comps.iter().map(|c| c.1).collect();
        // let idx0: Vec<i32> = comps.iter().map(|c| c.2).collect();

        // apply same delta as encoder
        let mut comps_delta = comps.clone();
        RansEnc::delta_encode(&mut comps_delta);

        let escapes = comps_delta.iter().filter(|c| c.3.is_some()).count();
        let escape_rate = escapes as f64 / n as f64;

        // Δexp stats
        let mut exp_zero = 0usize;
        let mut exp_abs_sum = 0f64;
        let mut exp_abs_sq_sum = 0f64;

        // Δidx stats (non-escape only)
        let mut idx_count = 0usize;
        let mut idx_zero = 0usize;
        let mut idx_abs_sum = 0f64;
        let mut idx_abs_sq_sum = 0f64;

        // top frequencies for Δidx (non-escape)
        let mut freq: HashMap<i32, u32> = HashMap::new();

        for c in &comps_delta {
            // Δexp always exists
            let de = c.1 as i32;
            if de == 0 { exp_zero += 1; }
            let ade = (de.abs()) as f64;
            exp_abs_sum += ade;
            exp_abs_sq_sum += ade * ade;

            // Δidx only counted when not escape
            if c.3.is_none() {
                idx_count += 1;
                let di = c.2;
                if di == 0 { idx_zero += 1; }
                let adi = (di.abs()) as f64;
                idx_abs_sum += adi;
                idx_abs_sq_sum += adi * adi;
                *freq.entry(di).or_insert(0) += 1;
            }
        }

        let exp_zero_pct = (exp_zero as f64 / n as f64) * 100.0;
        let exp_mean_abs = exp_abs_sum / n as f64;
        let exp_std_abs = ((exp_abs_sq_sum / n as f64) - exp_mean_abs * exp_mean_abs).max(0.0).sqrt();

        let (idx_zero_pct, idx_mean_abs, idx_std_abs) = if idx_count == 0 {
            (0.0, 0.0, 0.0)
        } else {
            let z = (idx_zero as f64 / idx_count as f64) * 100.0;
            let m = idx_abs_sum / idx_count as f64;
            let s = ((idx_abs_sq_sum / idx_count as f64) - m * m).max(0.0).sqrt();
            (z, m, s)
        };

        // top 5 most common Δidx values
        let mut items: Vec<(i32, u32)> = freq.into_iter().collect();
        items.sort_by_key(|&(_k, v)| std::cmp::Reverse(v));
        let top5: Vec<(i32, u32)> = items.into_iter().take(5).collect();

        println!(
            "stats: N={} escape={:.3}% | Δexp: zero={:.1}% mean|.|={:.3} std|.|={:.3} | Δidx(non-esc): zero={:.1}% mean|.|={:.1} std|.|={:.1} top5={:?}",
            n,
            escape_rate * 100.0,
            exp_zero_pct,
            exp_mean_abs,
            exp_std_abs,
            idx_zero_pct,
            idx_mean_abs,
            idx_std_abs,
            top5
        );
    }
    pub fn new(buffer_size: usize, encode: &'a Vec<f32>, step: f32, err: i32, mant_scale: u32) -> Self {
        let encoder = B64RansEncoderMulti::new(buffer_size); // recommend 1MiB starting internal buffer for 512KB blocks (double block size)
        let mant_alph_size = (((1 << 23) / step as usize) * 2) + 1;
        Self {
            encode,
            step,
            mant_alph_size,
            max_err: err,
            sign_context: RansEncContext::new(SIGN_ALPH_SIZE, 0, 8),
            exponent_context: RansEncContext::new(EXP_ALPH_SIZE, (EXP_ALPH_SIZE / 2) as i32, 12),
            mantissa_context: RansEncContext::new(mant_alph_size + 1, (mant_alph_size / 2) as i32, mant_scale ),
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

    pub fn encode_values(&mut self ) -> (Vec<u8>, (Vec<u8>, u32)) {
        let mut raw_symbols: Vec<u8> = vec![];

        //complete forward pass concurrently for the 3 parts.
        let mut components  = self.forward_pass();


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
                    self.encoder.put_at(ENC_MANTISSA_CHANNEL,&mant_symbols[self.mant_alph_size]);
                    raw_symbols.append(&mut comp.3.unwrap().to_ne_bytes().to_vec());
                } else {
                    self.encoder.put_at(ENC_MANTISSA_CHANNEL,&mant_symbols[self.mantissa_context.shift_idx(comp.2)]);
                }
            }


        }
        self.encoder.flush_all();

        (self.encoder.data().to_owned(), RansEnc::encode_raw(&mut raw_symbols))

    }

    fn encode_raw(raw_bytes: &mut  Vec<u8>) -> (Vec<u8>, u32) {
        let raw_len = raw_bytes.len() as u32;

        if raw_len == 0 { return (vec![], raw_len); }

        let mut raw_context = RansEncContext::new(EXP_ALPH_SIZE / 2, 0, 10);
        raw_context.build_snapshot();

        //forward pass
        for (i, &byte) in raw_bytes.iter().enumerate() {
            raw_context.increment_freq(byte as usize, i);
        }

        let mut encoder = B64RansEncoder::new(raw_bytes.len() * 2);

        let mut r_idx = 0;
        if let Some(idx) = raw_context.rescale_location.pop() { r_idx = idx; }
        let mut symbols = raw_context.snapshots.pop().unwrap_or_else(|| panic!("Failed to get raw context snapshot"));

        //backward pass
        for i in (0..raw_bytes.len()).rev() {
            if i == r_idx && i != 0 {
                if let Some(idx) = raw_context.rescale_location.pop() { r_idx = idx; }
                symbols = raw_context.snapshots.pop().unwrap_or_else(|| panic!("Failed to get raw context snapshot"));
            }

            if let Some(byte) = raw_bytes.pop() {
                encoder.put(&symbols[byte as usize]);
            }
        }

        encoder.flush_all();


        (encoder.data().to_owned(), raw_len)
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

    fn forward_pass(&mut self) -> Vec<(bool, i16, i32, Option<u32>)> {
        self.sign_context.build_snapshot();
        self.exponent_context.build_snapshot();
        self.mantissa_context.build_snapshot();

        //1. Componentize float and compute quantized mantissa idx
        let mut components = self.componentize_and_quantize(self.encode);

        //NOTE! debug code:
        // let quantized = self.rebuild_floats(&components);

        //2. Delta encode exponent and mantissa idx
        RansEnc::delta_encode(&mut components);


        //3. emulate decoder forward pass and caching
        for (i,comp)  in components.iter().enumerate() {

            self.sign_context.increment_freq(comp.0 as usize, i);
            self.exponent_context.increment_freq(self.exponent_context.shift_idx(comp.1 as i32),i);

            if comp.3 == None {
                self.mantissa_context.increment_freq(self.mantissa_context.shift_idx(comp.2), i);
            }
        };


        components

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

    fn componentize_and_quantize(&self, vals: &Vec<f32>) -> Vec<(bool, i16, i32, Option<u32>)> {
        let mut res : Vec<(bool, i16, i32, Option<u32>)> = Vec::with_capacity(vals.len());
        for val in vals {
            let mut components : (bool, i16, i32, Option<u32>);

            let bits = val.to_bits();

            //extract sign, exponent, mantissa
            let sign : bool = ((bits >> 31) & 1) != 0;
            let exp = ((bits >> 23) & 0xFF) as i16;
            let mant = bits & 0x7F_FFFF;

            //compute nearest quantization step
            //NOTE: I will delta code my quantization idx NOT the mantissa.
            let qm_idx = (mant as f32 / self.step).round() as i32;

            components = (sign, exp, qm_idx, None);

            //check if the quantized mantissa is in range
            if (mant as i32 - (qm_idx* self.step as i32)).abs() >= self.max_err {
                components.3 = Some(mant);
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
    scale_bit: u32,
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
    pub fn new(alphabet_len: usize, shift_range: i32, scale_bit: u32) -> Self {
        let context = Context::new(alphabet_len, scale_bit);
        Self { context, symbols: vec![], freq_to_symbol: vec![], alphabet_len, shift_range, scale_bit }
    }

    pub fn shift_idx(&self, symbol: usize) -> i32 {
        symbol as i32 - self.shift_range
    }

    pub fn increment_freq(&mut self, symbol: usize) {
        self.context.increment_freq(symbol);
    }


    pub fn rebuild_histogram(&mut self) {
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
        let total_freq = 1 << self.scale_bit;


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

const DEC_SIGN_CHANNEL: usize = 2;
const DEC_EXPONENT_CHANNEL: usize = 1;
const DEC_MANTISSA_CHANNEL: usize = 0;
pub struct RansDec<'a> {
    sign_context: RansDecContext,
    exponent_context: RansDecContext,
    mantissa_context: RansDecContext,
    decoder: B64RansDecoderMulti<'a,3>,
    raw_bytes: Vec<u8>,
    step: f32,
    mant_alph_size: usize,
    mant_scale: u32,
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
    pub fn new(code_data: &'a mut [u8], raw: &'a mut Vec<u8>, raw_len: u32, step: f32, mant_scale: u32) -> Self {
        let mant_alph_size = (((1 << 23) / step as usize) * 2) + 1;
        let raw_bytes = RansDec::decode_raw(raw, raw_len);
        Self {
            sign_context: RansDecContext::new(SIGN_ALPH_SIZE, 0, 8),
            exponent_context: RansDecContext::new(EXP_ALPH_SIZE, (EXP_ALPH_SIZE / 2) as i32, 12),
            mantissa_context: RansDecContext::new(mant_alph_size + 1, (mant_alph_size / 2) as i32, mant_scale),
            decoder: B64RansDecoderMulti::new(code_data),
            raw_bytes,
            step,
            mant_alph_size,
            mant_scale,
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

        for _ in 0..length {
            let mut component : (bool, i16, i32, Option<u32>);

            let sign_cum_freq = self.decoder.get_at(DEC_SIGN_CHANNEL, 8);
            let exp_cum_freq = self.decoder.get_at(DEC_EXPONENT_CHANNEL, 12);
            let mant_cum_freq = self.decoder.get_at(DEC_MANTISSA_CHANNEL, self.mant_scale);

            let sign_symbol = self.sign_context.freq_to_symbol[sign_cum_freq as usize];
            let exp_symbol = self.exponent_context.freq_to_symbol[exp_cum_freq as usize];
            let mant_symbol = self.mantissa_context.freq_to_symbol[mant_cum_freq as usize];


            self.sign_context.increment_freq(sign_symbol);
            self.exponent_context.increment_freq(exp_symbol);

            component = ((sign_symbol & 1) != 0, self.exponent_context.shift_idx(exp_symbol) as i16, self.mantissa_context.shift_idx(mant_symbol), None );

            if mant_symbol < self.mant_alph_size {
                self.mantissa_context.increment_freq(mant_symbol);
            } else {
                let val = u32::from_ne_bytes(self.raw_bytes[self.raw_bytes.len() - 4..].try_into().unwrap());
                self.raw_bytes.drain(self.raw_bytes.len() - 4..);
                component.2 = (val as f32 / self.step).round() as i32;
                component.3 = Some(val);
            }

            components.push(component);


            self.decoder.advance_step_at(DEC_SIGN_CHANNEL, &self.sign_context.symbols[sign_symbol], 8);
            self.decoder.advance_step_at(DEC_EXPONENT_CHANNEL,&self.exponent_context.symbols[exp_symbol], 12);
            self.decoder.advance_step_at(DEC_MANTISSA_CHANNEL,&self.mantissa_context.symbols[mant_symbol], self.mant_scale);
            self.decoder.renorm_all();

            self.sign_context.rebuild_histogram();
            self.exponent_context.rebuild_histogram();
            self.mantissa_context.rebuild_histogram();

        }

        RansDec::delta_decode(&mut components);

        self.rebuild_floats(&components)

    }

    fn decode_raw(raw_bytes: &'a mut [u8], raw_len: u32) -> Vec<u8> {

        if raw_len == 0 { return Vec::new(); }

        let mut res : Vec<u8> = Vec::with_capacity(raw_len as usize);

        let mut decoder = B64RansDecoder::new(raw_bytes);
        let scale_bits = 10;
        let mut decode_context = RansDecContext::new(EXP_ALPH_SIZE / 2, 0, scale_bits);

        decode_context.build_inverse_freq_table();

        for _ in 0..raw_len as usize {
            let cum_freq = decoder.get(scale_bits);
            let symbol = decode_context.freq_to_symbol[cum_freq as usize];
            decode_context.increment_freq(symbol);

            res.push(symbol as u8);

            decoder.advance(&decode_context.symbols[symbol], scale_bits);

            decode_context.rebuild_histogram();
        }

        res
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

    fn rebuild_floats(&self, components: &Vec<(bool, i16, i32, Option<u32>)>) -> Vec<f32> {
        let mut res : Vec<f32> = Vec::with_capacity(components.len());
        for comp in components {
            let mant : i32 = match comp.3 != None {
                true => {
                    comp.3.unwrap() as i32
                },
                false => comp.2 * self.step as i32};
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
