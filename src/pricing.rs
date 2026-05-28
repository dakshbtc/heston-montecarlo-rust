use rand::{SeedableRng, rngs::SmallRng};
use rayon::prelude::*;

use crate::models::HestonParams;
use crate::option::OptionParams;
use crate::simulation::{McConfig, simulate_path_antithetic};

/// Price a European option under the Heston model via Monte Carlo.
///
/// Optimisations applied:
///   - **Antithetic variates**: each RNG draw produces a normal path and its
///     mirror (`-z`), halving estimator variance at negligible extra cost.
///   - **Rayon parallelism**: paths are distributed across all CPU cores.
///   - **SmallRng (Xoshiro256++)**: ~2–3× faster than the default ChaCha RNG;
///     each thread gets its own seed derived from `seed ^ path_index` so
///     results are still reproducible.
///
/// Returns `(price, std_error)` where `std_error` is the 1σ standard error.
pub fn price_european(
    heston: &HestonParams,
    option: &OptionParams,
    config: &McConfig,
    seed: u64,
) -> (f64, f64) {
    let discount = (-option.r * option.t).exp(); // e^(-rT)

    // Each iteration produces two payoffs (normal + antithetic).
    // We run n_paths/2 iterations so total effective paths == n_paths.
    let half = config.n_paths / 2;

    let (payoff_sum, payoff_sq_sum) = (0..half)
        .into_par_iter()
        .map(|i| {
            // Per-thread RNG seeded deterministically from the master seed.
            let mut rng = SmallRng::seed_from_u64(seed ^ (i as u64).wrapping_mul(6364136223846793005).wrapping_add(1));

            let (s1, s2) = simulate_path_antithetic(heston, option, config, &mut rng);

            let payoff = |s: f64| -> f64 {
                if option.is_call {
                    (s - option.strike).max(0.0)
                } else {
                    (option.strike - s).max(0.0)
                }
            };

            // Average the two antithetic payoffs into a single low-variance sample.
            let p = (payoff(s1) + payoff(s2)) * 0.5;
            (p, p * p)
        })
        .reduce(|| (0.0, 0.0), |(a1, b1), (a2, b2)| (a1 + a2, b1 + b2));

    let n = half as f64;
    let mean_payoff = payoff_sum / n;                                    // E[Φ] = (1/N) Σ Φᵢ
    let variance = (payoff_sq_sum / n) - (mean_payoff * mean_payoff);   // Var  = E[Φ²] - E[Φ]²
    let std_error = (variance / n).sqrt() * discount;                   // SE   = e^(-rT) · √(Var/N)
    let price: f64 = mean_payoff * discount;                            // V₀   = e^(-rT) · E[Φ]

    (price, std_error)
}
