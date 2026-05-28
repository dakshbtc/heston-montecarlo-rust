mod models;
mod option;
mod pricing;
mod simulation;
mod viz;

use std::time::Instant;

use models::HestonParams;
use option::OptionParams;
use pricing::price_european;
use simulation::McConfig;

fn main() {
    let heston = HestonParams {
        kappa: 2.0,  // variance mean-reverts twice per year
        theta: 0.04, // long-run variance → 20% vol  (√0.04 = 0.20)
        sigma: 0.3,  // vol of vol = 30%
        rho: -0.7,   // negative correlation (leverage effect)
        v0: 0.04,    // start at the long-run level
    };

    let option = OptionParams {
        s0: 100.0,
        strike: 100.0,
        r: 0.05,
        t: 1.0,
        is_call: true,
    };

    let config = McConfig {
        n_paths: 100_000,
        n_steps: 252, // one step per trading day
    };

    println!("=== Heston Model Monte Carlo Pricer ===");
    println!("  optimisations: rayon parallelism  |  SmallRng (Xoshiro256++)  |  antithetic variates\n");

    println!("Heston Parameters:");
    println!("  kappa (mean reversion) : {}", heston.kappa);
    println!("  theta (long-run var)   : {} ({:.1}% vol)", heston.theta, heston.theta.sqrt() * 100.0);
    println!("  sigma (vol of vol)     : {}", heston.sigma);
    println!("  rho   (correlation)    : {}", heston.rho);
    println!("  v0    (initial var)    : {} ({:.1}% vol)\n", heston.v0, heston.v0.sqrt() * 100.0);

    println!("Option Parameters:");
    println!("  Spot   S0 : {}", option.s0);
    println!("  Strike K  : {}", option.strike);
    println!("  Rate   r  : {}%", option.r * 100.0);
    println!("  Expiry T  : {} year(s)", option.t);
    println!("  Type      : {}\n", if option.is_call { "Call" } else { "Put" });

    println!("Simulation: {} paths × {} steps  ({} antithetic pairs)\n",
        config.n_paths, config.n_steps, config.n_paths / 2);

    // ── Pricing ───────────────────────────────────────────────────────────────

    let t0 = Instant::now();
    let (call_price, call_se) = price_european(&heston, &option, &config, 42);
    let elapsed = t0.elapsed();

    println!("Call Price : {:.4}  ±  {:.4} (1σ std error)", call_price, call_se);

    // Put-call parity: C - P = S₀ - K·e^(-rT)  →  P = C - S₀ + K·e^(-rT)
    let parity_put = call_price - option.s0 + option.strike * (-option.r * option.t).exp();
    println!("Put  Price : {:.4}  (via put-call parity)\n", parity_put);

    let put_option = OptionParams { is_call: false, ..option };
    let (put_price, put_se) = price_european(&heston, &put_option, &config, 42);
    println!("Put  Price : {:.4}  ±  {:.4} (direct MC)", put_price, put_se);

    let parity_diff = (parity_put - put_price).abs();
    println!("\nPut-call parity error : {:.6}  (should be near zero)", parity_diff); 
    println!("Wall time             : {:.2?}  (use --release for full speed)\n", elapsed);

    // ── Visualisations ────────────────────────────────────────────────────────

    println!("Generating visualisations…");

    let t1 = Instant::now();

    viz::plot_paths(&heston, &option, &config, 42, "paths.png")
        .expect("failed to write paths.png");

    viz::plot_convergence(&heston, &option, &config, 42, call_price, "convergence.png")
        .expect("failed to write convergence.png");

    println!("  paths.png        — {} sample paths + ensemble mean + strike",
        viz::N_DISPLAY);
    println!("  convergence.png  — running price estimate converging to {:.4}", call_price);
    println!("  viz time         : {:.2?}", t1.elapsed());
}
