/**
 * File Name: rans_coding
 * Description: This file implements a block-adaptive rANS coder and decoder
 * Date Created: 11/05/2025
 * Date Last Modified: 11/05/2025
 */

use rans::b64_encoder::{B64RansEncSymbol, B64RansEncoder, B64RansEncoderMulti};
use rans::{RansEncSymbol, RansEncoder, RansEncoderMulti, RansDecoder, RansDecSymbol};
use rans::b64_decoder::{B64RansDecoder, B64RansDecSymbol};

const NUM_SYMBOLS: usize = 100;
// const SHIFT_RANGE: i32 = NUM_SYMBOLS as i32 / 2;
const SCALE_BIT: u32 = 14;

struct Context {
    alphabet_len: usize,
    shift_range: i32,
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
        Self{ alphabet_len, shift_range: alphabet_len as i32 / 2,  freq: vec![1; alphabet_len  + 1],  total_freq: alphabet_len + 1 }
    }

    //TODO: consider whether we should "adapt" aka increment frequency of the escape symbol
    //may not matter much. May be better determined with testing
    //currently we do not increment the escape character
    pub fn increment_freq(&mut self, idx: usize) {
            self.freq[idx] += 1;
            self.total_freq += 1;
    }

    // pub fn decrement_freq(&mut self, idx: usize) {
    //     self.freq[idx] -= 1;
    //     self.total_freq -= 1;
    // }

    pub fn get_freq_array(&self) -> &[u16] {
       &self.freq
    }

    pub fn rebuild_histogram(&self) -> bool {
       self.total_freq == (1 << SCALE_BIT)
    }

    /**Shift range
     * Shifts a signed integer to the unsigned alphabet range starting at 0
     *
     * return: (bool, usize)
     * bool -  false if out of range (signals escape character should be used)
     * usize - the shifted unsigned integer index
     */
    pub fn shift_range(&self, symbol: i32) -> (bool, usize) {
        if symbol < -self.shift_range || symbol >= self.shift_range {
           //if out of range, return escape character
            (false, self.alphabet_len)
        } else {
            (true, (symbol + self.shift_range) as usize)
        }
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

pub struct RansEncContext {
    context: [Context; 3],
    encoder: B64RansEncoderMulti<3>,
    snapshots: Vec<Vec<B64RansEncSymbol>>,
    //consider turning this into a single flat index, for now this is proof of concept
    //would probably have to communicate how many partitions there are with each data stream
    rescale_location: Vec<usize>


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
    pub fn new(buffer_size: usize ) -> Self {
        //TODO specify the correct NUM_SYMBOLS
        let context = [Context::new(NUM_SYMBOLS), Context::new(NUM_SYMBOLS), Context::new(NUM_SYMBOLS)];
        let encoder = B64RansEncoderMulti::new(buffer_size); // recommend 1MiB starting internal buffer for 512KB blocks (double block size)
        Self { context, encoder, snapshots: vec![], rescale_location: vec![]}
    }

    pub fn encode_values(&mut self, values: &Vec<i32> ) -> (Vec<u8>, Vec<u8>) {
        let mut raw_symbols: Vec<u8> = vec![];

        //TODO component-wise breakdown and delta-coding

        //complete forward pass concurrently for the 3 parts.
        self.forward_pass(values);


        println!("Beginning Backward Pass");
        let mut rescale = 0;
        if let Some(idx) = self.rescale_location.pop() {
            rescale = idx;
        }

        let mut symbols : Vec<B64RansEncSymbol> = self.snapshots.pop().unwrap_or_else(|| panic!("Failed to get context snapshot"));

        for (i, symbol) in values.iter().enumerate().rev() {
            match self.context.shift_range(*symbol) {
                (true, idx) => {
                    self.encoder.put_at(/*channel*/,&symbols[idx])
                },
                (false, idx) => {
                    println!("Escape symbol encoded due to out-of-range symbol: {:?}", symbol);
                    //encode escape symbol
                    self.encoder.put_at(/*channel*/, &symbols[idx]);

                    //flush encoder buffer and store
                    // self.encoder.flush_all();
                    // code.append(&mut self.encoder.data().to_owned());

                    //append the out-of-range symbol's raw bytes
                    //perhaps more could be done here to reduce size (hopefully this doesn't happen frequently)
                    raw_symbols.append(&mut symbol.to_ne_bytes().to_vec());

                    //reset encoder to pick off where left off
                    // self.encoder.reset();
                }
            }

            if i == rescale && i != 0 {
                // println!("rescaling at {:?}", i);
                if let Some(idx) = self.rescale_location.pop() {
                    rescale = idx;
                }
                symbols = self.snapshots.pop().unwrap_or_else(|| panic!("Failed to get context snapshot"));
            }
        }
        println!("Completed Backward Pass\n");

        self.encoder.flush_all();

        (self.encoder.data().to_owned(), raw_symbols)

    }

    fn component_wise_breakdown(&self, values: &Vec<f32>) -> (Vec<u8>, Vec<u8>, Vec<u32>) {
        let mut signs : Vec<u8> = Vec::with_capacity(values.len());
        let mut exponents : Vec<u8> = Vec::with_capacity(values.len());
        let mut mantissa_bits = Vec::with_capacity(values.len());

        //first value not delta coded
        if let Some(val) = values.first() {
            let bits = val.to_bits();
            signs.push(((bits >> 31) & 0x1) as u8);
            exponents.push(((bits >> 23) & 0xFF) as u8);
            mantissa_bits.push(bits & 0x7F_FFFF );
        }

        //delta code exponent and mantissa of remaining values
        if values.len() > 1 {
            for i  in 1..values.len() {
                let bits = values[i].to_bits();
                signs.push(((bits >> 31) & 0x1) as u8);
                exponents.push(
                    ((bits >> 23) & 0xFF) as u8 - exponents[i - 1]
                );
                mantissa_bits.push(bits & 0x7F_FFFF - mantissa_bits[i - 1]);
            };
        }

            (signs, exponents, mantissa_bits)
    }



    fn forward_pass(&mut self, values: &Vec<i32>){
        println!("Beginning Forward Pass");

        self.build_snapshot();
        for (i, val) in values.iter().enumerate() {
            if self.context.rebuild_histogram() /*&& (values.len() - (i + 1)) > (1 << SCALE_BIT)*/ {
                self.build_snapshot();
                self.context.rescale_model();
                self.rescale_location.push(i);
            }
            match self.context.shift_range(*val) {
                (true, idx) => {
                    self.context.increment_freq(idx);
                },
                //this skips incrementing frequency if out of range
                _ => {
                  //   println!("Skipping freq increment");
                },
            }
        }

        println!("Completed Forward Pass");
    }

    fn build_snapshot(&mut self) {
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

        // println!("New histogram: {:?}", res);

        self.snapshots.push(res);
    }

}

pub struct RansDec<'a> {
    context: Context,
    decoder: B64RansDecoder<'a>,
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
impl<'a> RansDec<'a> {
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
