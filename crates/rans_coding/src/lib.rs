// we will use a fenwick tree in order to represent my cumulative frequencies
// this will begin with a single, contigious alphabet across the range [-100,100]
// normalized to [0,200]. The value will be the index into the fenwick tree
use fenwick::array::{update, prefix_sum};
pub struct Context {
    pub fenwick_tree : [i32; 200],
}

impl Context {
    pub fn new() -> Self {
        let mut fenwick_tree = [0; 200];
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

    fn init_all_ones(tree: &mut [i32; 200]) {
        for (i, slot) in tree.iter_mut().enumerate() {
            let idx = i + 1;
            let lsb = 1 << (idx.trailing_zeros());
            *slot = lsb;
        }
    }
}

