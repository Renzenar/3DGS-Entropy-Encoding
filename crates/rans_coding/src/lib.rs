/**
 * File Name: rans_coding
 * Description: This file implements a block-adaptive rANS coder and decoder
 * Date Created: 11/05/2025
 * Date Last Modified: 11/05/2025
 */

use rans::b64_encoder::{B64RansEncSymbol, B64RansEncoder};
use rans::{RansEncSymbol, RansEncoder, RansEncoderMulti, RansDecoder, RansDecSymbol};
use rans::b64_decoder::{B64RansDecoder, B64RansDecSymbol};

const NUM_SYMBOLS: usize = 400;
const SHIFT_RANGE: i32 = NUM_SYMBOLS as i32 / 2;
const SCALE_BIT: u32 = 14;

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
    pub fn new(len: usize) -> Self {
        Self{ freq: vec![1; len],  total_freq: len }
    }

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

    pub fn shift_range(symbol: i32) -> usize {
        if symbol < -SHIFT_RANGE || symbol >= SHIFT_RANGE {
            println!("symbol error: {:?}", symbol);
            panic!("symbol out of range") }
        (symbol + SHIFT_RANGE) as usize
    }

    /**Description: rescale_model scales each index in the array by a factor of 2. If it scales to
     * 0, it will resolve to 1. Total_freq is updated with the new scale.
     */
    pub fn rescale_model(&mut self) {
        let mut total = 0u32;
        self.freq.iter_mut().for_each(|x| {
            *x >>= 1;
            if *x == 0 {
                *x = 1;
            }
            total += *x as u32;
        });
        self.total_freq = total as usize;
    }
}

pub struct RansEnc {
    context: Context,
    encoder: B64RansEncoder,
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
impl RansEnc {
    pub fn new(buffer_size: usize ) -> Self {
        let context = Context::new(NUM_SYMBOLS);
        let encoder = B64RansEncoder::new(buffer_size); // recommend 1MiB starting internal buffer for 512KB blocks (double block size)
        Self { context, encoder, snapshots: vec![], rescale_location: vec![]}
    }

    pub fn encode_values(&mut self, values: &Vec<i32>) -> Vec<u8> {
        println!("Beginning Forward Pass");

        self.forward_pass(values);

        println!("Completed Forward Pass");

        println!("Beginning Backward Pass");
        let mut rescale = 0;
        if let Some(idx) = self.rescale_location.pop() {
            rescale = idx;
        }

        let mut symbols : Vec<B64RansEncSymbol> = self.snapshots.pop().unwrap_or_else(|| panic!("Failed to get context snapshot"));

        for (i, symbol) in values.iter().enumerate().rev() {
            self.encoder.put(&symbols[Context::shift_range(*symbol)]);

            //this very well may have to happen before the symbol is encoded
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

        self.encoder.data().to_owned()

    }

    fn forward_pass(&mut self, values: &Vec<i32>){
        self.build_snapshot();
        for (i, val) in values.iter().enumerate() {
            if self.context.rebuild_histogram() /*&& (values.len() - (i + 1)) > (1 << SCALE_BIT)*/ {
                self.build_snapshot();
                self.context.rescale_model();
                self.rescale_location.push(i);
            }
            self.context.increment_freq(Context::shift_range(*val));
        }
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
    pub fn new(data: &'a mut [u8]) -> Self {
        let context = Context::new(NUM_SYMBOLS);
        let decoder = B64RansDecoder::new(data);
        Self { context, decoder, symbols: vec![], freq_to_symbol: vec![] }
    }

    pub fn decode_values(&mut self, length: usize) -> Vec<i32> {
        let mut res = Vec::with_capacity(length);
        self.build_inverse_freq_table();

        println!("\nBeginning Decoding");
        for _ in 0..length {
            let cum_freq = self.decoder.get(SCALE_BIT);

            let symbol = self.freq_to_symbol[cum_freq as usize];


            res.push(symbol as i32 - SHIFT_RANGE);


            self.decoder.advance(&self.symbols[symbol], SCALE_BIT);

            self.context.increment_freq(symbol);

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
