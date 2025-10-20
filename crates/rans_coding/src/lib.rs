// we will use a fenwick tree in order to represent my cumulative frequencies
// this will begin with a single, contigious alphabet across the range [-100,100]
// normalized to [0,200]. The value will be the index into the fenwick tree
// SCALE BITS = 8 (2^8 == 256 which is greater than our alphabet range)
use fenwick::array::{update, prefix_sum};
pub struct Context {
    pub fenwick_tree : [i32; 200],
}

const TREE_LEN: usize = 200;

impl Context {
    pub fn new() -> Self {
        let mut fenwick_tree = [0; TREE_LEN];
        Self::init_all_ones(&mut fenwick_tree);
        Self { fenwick_tree }
    }

    // cum(i) = prefix_sum(i -1) per rANS definition
    pub fn get_cum_freq(&self, symbol: u8) -> i32{
        if symbol == 0 {0} else {prefix_sum(&self.fenwick_tree, symbol as usize - 1)}
    }

    // freq(i) = prefix_sum(i) - prefix_sum(i-1) per rANS definition
    pub fn get_freq(&self, symbol: u8) -> i32{
        // if symbol == 0 {self.fenwick_tree[symbol as usize]} else {self.fenwick_tree[symbol as usize] - self.fenwick_tree[symbol as usize - 1]}
        if symbol == 0 {prefix_sum(&self.fenwick_tree, 0)} else {prefix_sum(&self.fenwick_tree, symbol as usize) - prefix_sum(&self.fenwick_tree, symbol as usize - 1)}
    }

    pub fn increment_freq(&mut self, symbol: u8){
        update(&mut self.fenwick_tree, symbol as usize, 1);
    }

    //This will use a bit lifting approach
    //start at the highest power of 2
    // - <= target?
    //     -if so keep that bit
    //     -if not reject that bit
    // - test the next power of 2
    pub fn get_symbol_from_cum_freq(&self, cum_freq: i32) -> i32 {
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

    //initializes everything to a frequency of 1
    fn init_all_ones(tree: &mut [i32; TREE_LEN]) {
        for (i, slot) in tree.iter_mut().enumerate() {
            let idx = i + 1;
            let lsb = 1 << (idx.trailing_zeros());
            *slot = lsb;
        }
    }
}



