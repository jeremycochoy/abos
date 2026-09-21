//! A fast deterministic hasher for integer keys (issue #10).
//!
//! The book maps hold `u64` order ids. The std `SipHash` costs a visible
//! share of a replay; this multiply-and-rotate hash (the rustc-hash
//! algorithm) is a few instructions. No dependency, no per-process seed.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

/// Multiply-and-rotate hasher for integer keys.
#[derive(Default)]
pub struct FxHasher {
    hash: u64,
}

impl FxHasher {
    fn mix(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(SEED);
    }
}

impl Hasher for FxHasher {
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut word = [0u8; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            self.mix(u64::from_le_bytes(word));
        }
    }

    fn write_u64(&mut self, n: u64) {
        self.mix(n);
    }

    fn write_usize(&mut self, n: usize) {
        self.mix(n as u64);
    }

    fn finish(&self) -> u64 {
        self.hash
    }
}

/// A `HashMap` keyed with [`FxHasher`].
pub type FxHashMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;

pub type FxHashSet<K> = std::collections::HashSet<K, BuildHasherDefault<FxHasher>>;
