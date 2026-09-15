//! Stable hashing for affinity keys and rendezvous ranking.
//!
//! `std::collections::hash_map::DefaultHasher` is explicitly not stable across
//! Rust releases, and `RandomState` is seeded per process. Either would make an
//! affinity key mean something different after a restart or a toolchain bump,
//! which would silently destroy comparability between two runs of the same
//! experiment. FNV-1a is fixed by its specification, trivially auditable, and
//! fast enough: the router hashes at most a few hundred token IDs per request
//! against a serving path that costs tens of milliseconds.

/// FNV-1a, 64-bit. Constants from the FNV specification.
#[inline]
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut h = OFFSET;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(PRIME);
    }
    h
}

/// The hash of a request's routable prompt prefix.
///
/// Wrapped in a newtype rather than passed as a bare `u64` so it cannot be
/// confused with a replica index or a request id at a call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AffinityKey(pub u64);

impl AffinityKey {
    /// Hash the first `n` token IDs of a prompt.
    ///
    /// Token IDs, not decoded text: the engine's prefix cache keys on the exact
    /// token sequence, so hashing anything else risks grouping requests that the
    /// engine will not treat as sharing a prefix (and separating ones it will).
    /// Encoded little-endian so the key is reproducible across machines.
    pub fn from_token_ids(ids: &[i64], n: usize) -> Self {
        let take = n.min(ids.len());
        let mut buf = Vec::with_capacity(take * 8);
        for id in &ids[..take] {
            buf.extend_from_slice(&id.to_le_bytes());
        }
        Self(fnv1a64(&buf))
    }

    /// Hash the first `n` *characters* of a text prompt.
    ///
    /// Used for `/v1/chat/completions`, where the router sees text and the
    /// tokenizer lives inside the replica. This is an approximation of the token
    /// prefix and is documented as such: character `n` is not token `n`. It is
    /// still a correct affinity key (equal prefixes hash equally), just a
    /// coarser one. Truncation respects char boundaries so the key is
    /// well-defined for multi-byte text.
    pub fn from_text(text: &str, n: usize) -> Self {
        let end = text
            .char_indices()
            .nth(n)
            .map(|(i, _)| i)
            .unwrap_or(text.len());
        Self(fnv1a64(text[..end].as_bytes()))
    }
}

/// Rendezvous (highest-random-weight) score for a (key, replica) pair.
///
/// Rendezvous hashing rather than a hash ring, for one reason that matters to
/// this experiment: when a replica is ejected, only the keys that mapped to *it*
/// move, and they redistribute evenly over the survivors. A naive
/// `key % n_healthy` remaps essentially every key the moment the healthy count
/// changes, which would flush every replica's prefix cache at once during the
/// failure-mode experiment and make the recovery measurement a measurement of
/// mass cache invalidation instead of routing.
///
/// The replica is identified by its *stable index*, not by its position in the
/// healthy list, so scores do not shift when the healthy set changes.
#[inline]
pub fn rendezvous_score(key: AffinityKey, replica_index: usize) -> u64 {
    let mut buf = [0u8; 16];
    buf[..8].copy_from_slice(&key.0.to_le_bytes());
    buf[8..].copy_from_slice(&(replica_index as u64).to_le_bytes());
    fnv1a64(&buf)
}

/// Rank the given replica indices by descending rendezvous score for `key`.
///
/// Ties break on the lower index so the ordering is total and deterministic.
pub fn rank_by_affinity(key: AffinityKey, indices: &[usize]) -> Vec<usize> {
    let mut scored: Vec<(u64, usize)> = indices
        .iter()
        .map(|&i| (rendezvous_score(key, i), i))
        .collect();
    scored.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    scored.into_iter().map(|(_, i)| i).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn fnv1a_matches_published_vectors() {
        // Reference values from the FNV-1a 64-bit specification.
        assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a64(b"foobar"), 0x8526_7b25_6108_9d4d);
    }

    #[test]
    fn equal_prefixes_hash_equally_and_longer_tails_do_not_matter() {
        let a: Vec<i64> = (0..1000).collect();
        let mut b: Vec<i64> = (0..1000).collect();
        // Diverge well past the hashed prefix.
        for x in b.iter_mut().skip(512) {
            *x = 999_999;
        }
        assert_eq!(
            AffinityKey::from_token_ids(&a, 512),
            AffinityKey::from_token_ids(&b, 512),
            "requests sharing a 512-token prefix must land on the same key"
        );
    }

    #[test]
    fn divergence_inside_the_hashed_prefix_changes_the_key() {
        let a: Vec<i64> = (0..1000).collect();
        let mut b = a.clone();
        b[511] = -1;
        assert_ne!(
            AffinityKey::from_token_ids(&a, 512),
            AffinityKey::from_token_ids(&b, 512)
        );
    }

    #[test]
    fn short_prompts_hash_their_whole_length_without_panicking() {
        let short: Vec<i64> = vec![1, 2, 3];
        let k = AffinityKey::from_token_ids(&short, 512);
        assert_eq!(k, AffinityKey::from_token_ids(&[1, 2, 3], 512));
    }

    #[test]
    fn text_keys_respect_char_boundaries() {
        // 4 multi-byte chars; taking 2 must not slice mid-codepoint.
        let s = "αβγδ";
        let k = AffinityKey::from_text(s, 2);
        assert_eq!(k, AffinityKey::from_text("αβ", 2));
        assert_eq!(AffinityKey::from_text(s, 999), AffinityKey::from_text(s, 4));
    }

    #[test]
    fn rendezvous_ranking_is_a_total_order_over_the_input() {
        let idx = vec![0usize, 1, 2, 3];
        let order = rank_by_affinity(AffinityKey(12345), &idx);
        assert_eq!(order.len(), 4);
        assert_eq!(
            order.iter().copied().collect::<HashSet<_>>(),
            idx.into_iter().collect()
        );
    }

    #[test]
    fn removing_a_replica_only_remaps_that_replica_s_keys() {
        // The property that motivates rendezvous over modulo hashing.
        let all = vec![0usize, 1, 2, 3];
        let survivors = vec![0usize, 1, 3]; // 2 ejected
        let mut moved = 0usize;
        let mut owned_by_two = 0usize;
        const N: u64 = 20_000;
        for k in 0..N {
            let key = AffinityKey(k.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let before = rank_by_affinity(key, &all)[0];
            let after = rank_by_affinity(key, &survivors)[0];
            if before == 2 {
                owned_by_two += 1;
            }
            if before != after {
                moved += 1;
            }
        }
        assert_eq!(
            moved, owned_by_two,
            "exactly the ejected replica's keys move, and no others"
        );
        // And its share was roughly a quarter, i.e. the hash spreads.
        let frac = owned_by_two as f64 / N as f64;
        assert!(
            (0.2..0.3).contains(&frac),
            "rendezvous share for 1 of 4 replicas was {frac}, expected ~0.25"
        );
    }

    #[test]
    fn keys_spread_roughly_evenly_over_replicas() {
        let idx = vec![0usize, 1, 2, 3, 4, 5, 6, 7];
        let mut counts = [0usize; 8];
        const N: u64 = 40_000;
        for k in 0..N {
            let key = AffinityKey(k.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            counts[rank_by_affinity(key, &idx)[0]] += 1;
        }
        let expected = N as f64 / 8.0;
        for (i, c) in counts.iter().enumerate() {
            let dev = (*c as f64 - expected).abs() / expected;
            assert!(
                dev < 0.10,
                "replica {i} took {c} keys, {dev:.3} off uniform"
            );
        }
    }

    #[test]
    fn ranking_is_stable_across_calls() {
        let idx = vec![0usize, 1, 2, 3];
        let key = AffinityKey(0xdead_beef);
        let a = rank_by_affinity(key, &idx);
        let b = rank_by_affinity(key, &idx);
        assert_eq!(a, b);
    }
}
