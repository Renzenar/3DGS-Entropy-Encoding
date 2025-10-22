# Reolve Rust Build Errors for the rANS crate (Manjaro Linux) 
I'll generalize this later

 1) Install the needed toolchain bits
sudo pacman -S --needed base-devel clang llvm llvm-libs

 2) Make sure libclang is visible (usually /usr/lib/libclang.so*)
ls /usr/lib/libclang.so*

 3) Export LIBCLANG_PATH so bindgen can find it
export LIBCLANG_PATH=/usr/lib

 4) Clean and rebuild this crate (or the whole workspace)
cargo clean -p ryg-rans-sys
cargo build -p rans_coding -vv
