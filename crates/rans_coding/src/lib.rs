// we will use a fenwick tree in order to represent my cumulative frequencies
// this will begin with a single, contigious alphabet across the range [-100,100]
// normalized to [0,200]. The value will be the index into the fenwick tree
// SCALE BITS = 8 (2^8 == 256 which is greater than our alphabet range)
use fenwick::array::{update, prefix_sum};
// use rans::{RansEncSymbol, RansEncoder};
use rans::b64_encoder::{B64RansEncSymbol, B64RansEncoder};
use rans::{RansEncSymbol, RansEncoder, RansEncoderMulti, RansDecoder, RansDecSymbol};
use rans::b64_decoder::{B64RansDecoder, B64RansDecSymbol};

const TREE_LEN: usize = 200;
const SCALE_BIT: u32 = 12;

pub struct Context {
    total_freq: usize,
    fenwick_tree : [i32; TREE_LEN],
}

impl Context {
    pub fn new() -> Self {
        let mut fenwick_tree = [0; TREE_LEN];
        Self::init_all_ones(&mut fenwick_tree);
        Self { total_freq: TREE_LEN, fenwick_tree }
    }

    pub fn norm_cum_freq(&self, symbol: usize) -> i32 {
        let total = 1 << SCALE_BIT;
        let total_freq = self.total_freq;
        let raw_cum_freq = self.get_cum_freq(symbol);

        (raw_cum_freq * total) / total_freq as i32
    }

    pub fn norm_freq(&self, symbol: usize) -> i32 {
        let symbol_cum_freq = self.norm_cum_freq(symbol + 1);
        let prev_cum_freq = self.norm_cum_freq(symbol);

        let res = symbol_cum_freq - prev_cum_freq;
        assert_ne!(res, 0);

        res
    }

    // cum(i) = prefix_sum(i -1) per rANS definition
    fn get_cum_freq(&self, symbol: usize) -> i32{
        if symbol == 0 {0} else {prefix_sum(&self.fenwick_tree, symbol - 1)}
    }


    // freq(i) = prefix_sum(i) - prefix_sum(i-1) per rANS definition
    // pub fn get_freq(&self, symbol: usize) -> i32{
    //     if symbol == 0 {prefix_sum(&self.fenwick_tree, 0)} else {prefix_sum(&self.fenwick_tree, symbol) - prefix_sum(&self.fenwick_tree, symbol - 1)}
    // }

    pub fn increment_freq(&mut self, symbol: usize){
        update(&mut self.fenwick_tree, symbol, 1);
        self.total_freq += 1;
        // overflow here shouldn't be an issue if the block size is controlled but may need to consider.
        // self.total_freq = self.total_freq.wrapping_add(1);
    }

    pub fn decrement_freq(&mut self, symbol: usize){
        update(&mut self.fenwick_tree, symbol, -1);
        self.total_freq -= 1;
    }

    pub fn get_symbol_from_norm_cum_freq(&self, norm_cum_freq: u32) -> i32 {
        let total = 1 << SCALE_BIT;

        let cum_freq = (norm_cum_freq * self.total_freq as u32) / total;

        self.get_symbol_from_cum_freq(cum_freq as i32)

    }

    //This will use a bit lifting approach
    //start at the highest power of 2
    // - <= target?
    //     -if so keep that bit
    //     -if not reject that bit
    // - test the next power of 2
    fn get_symbol_from_cum_freq(&self, cum_freq: i32) -> i32 {
        //! this must be set relative to fenwick tree length (the highest power of two)
        let mut bit = 128usize;

        let mut idx = 0usize;
        let mut acc = 0i32;

        while bit != 0 {
            let next = idx + bit;

            if next <= TREE_LEN {
                let block = self.fenwick_tree[next - 1];
                if acc + block <= cum_freq {
                    acc += block;
                    idx = next;
                }

            }
            bit >>= 1;
        }

        idx as i32
    }

    pub fn shift_range(symbol: i32) -> usize {
        if symbol < -100 || symbol >= 100 { panic!("symbol out of range") }
        (symbol + 100) as usize
    }

    //initializes everything to a frequency of 1
    fn init_all_ones(tree: &mut [i32; TREE_LEN]) {
        for (i, slot) in tree.iter_mut().enumerate() {
            let idx = i + 1;
            let lsb = 1 << (idx.trailing_zeros());
            *slot = lsb;
        }
    }
}

pub struct RansEnc {
    context: Context,
    encoder: B64RansEncoder,
}


impl RansEnc {
    pub fn new(buffer_size: usize ) -> Self {
        let context = Context::new();
        let encoder = B64RansEncoder::new(buffer_size); // recommend 1MiB starting internal buffer for 512KB blocks (double block size)
        Self { context, encoder /*, scale_bit: 12*/ }
    }

    pub fn encode_values(&mut self, values: &Vec<i32>) -> Vec<u8> {
        println!("Beginning Forward Pass");

        self.forward_pass(values);

        println!("Completed Forward Pass");

        println!("Beginning Backward Pass");
        for symbol in values.iter().rev() {
            // println!("Symbol: {}", *symbol);
            self.encoder.put(&B64RansEncSymbol::new(
                self.context.norm_cum_freq(Context::shift_range(*symbol)) as u32,
                self.context.norm_freq(Context::shift_range(*symbol)) as u32,
                SCALE_BIT,
            ));

            //decrement count
            self.context.decrement_freq(Context::shift_range(*symbol));
        }
        println!("Completed Backward Pass");

        self.encoder.flush_all();

        self.encoder.data().to_owned()

    }

    fn forward_pass(&mut self, values: &Vec<i32>){
        for symbol in values {
            self.context.increment_freq(Context::shift_range(*symbol));
        }
    }



}

pub struct RansDec<'a> {
    context: Context,
    decoder: B64RansDecoder<'a>,
}


impl<'a> RansDec<'a> {
    pub fn new(data: &'a mut [u8]) -> Self {
        let context = Context::new();
        let decoder = B64RansDecoder::new(data);
        Self { context, decoder }
    }

    pub fn decode_values(&mut self, length: usize) -> Vec<i32> {
        let mut res = vec!();

        for _ in 0..length {
            let norm_cum_freq = self.decoder.get(SCALE_BIT);

            let symbol = self.context.get_symbol_from_norm_cum_freq(norm_cum_freq);

            self.context.increment_freq(symbol as usize);

            res.push(symbol - 100);

            self.decoder.advance(&B64RansDecSymbol::new(
                norm_cum_freq,
                self.context.norm_freq(symbol as usize) as u32),
                                 SCALE_BIT,
            );
        }

        res
    }

    fn shift_range(&self, symbol: usize) -> i32 {
        symbol as i32 - 100
    }
}