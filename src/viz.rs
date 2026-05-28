use plotters::prelude::*;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use crate::models::HestonParams;
use crate::option::OptionParams;
use crate::simulation::{McConfig, simulate_path, simulate_path_record};

/// Number of sample paths drawn in the path plot.
pub const N_DISPLAY: usize = 250;

/// Plot N_DISPLAY sample Heston paths plus the ensemble mean and the strike
/// price as a horizontal reference line. Saves to `output` (PNG).
pub fn plot_paths(
    heston: &HestonParams,
    option: &OptionParams,
    config: &McConfig,
    seed: u64,
    output: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(42));
    let dt = option.t / config.n_steps as f64;

    let paths: Vec<Vec<f64>> = (0..N_DISPLAY)
        .map(|_| simulate_path_record(heston, option, config, &mut rng))
        .collect();

    // Y-axis range from 2nd–98th percentile of terminal prices, padded to
    // always include the strike.
    let mut finals: Vec<f64> = paths.iter().map(|p| *p.last().unwrap()).collect();
    finals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y_lo = (finals[N_DISPLAY * 2 / 100] * 0.88).min(option.strike * 0.78);
    let y_hi = (finals[N_DISPLAY * 98 / 100] * 1.12).max(option.strike * 1.22);

    let root = BitMapBackend::new(output, (1400, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!(
                "Heston MC — {} Sample Paths   \
                 S\u{2080}={:.0}, K={:.0}, T={:.1}yr, r={:.1}%,  \
                 \u{03BA}={}, \u{03B8}={}, \u{03C3}={}, \u{03C1}={}",
                N_DISPLAY,
                option.s0, option.strike, option.t, option.r * 100.0,
                heston.kappa, heston.theta, heston.sigma, heston.rho,
            ),
            ("sans-serif", 20).into_font(),
        )
        .margin(30)
        .x_label_area_size(50)
        .y_label_area_size(70)
        .build_cartesian_2d(0f64..option.t, y_lo..y_hi)?;

    chart
        .configure_mesh()
        .x_desc("Time (years)")
        .y_desc("Stock Price")
        .x_labels(10)
        .y_labels(10)
        .draw()?;

    // Individual paths — thin, translucent cornflower blue
    for path in &paths {
        chart.draw_series(LineSeries::new(
            path.iter().enumerate().map(|(i, &s)| (i as f64 * dt, s)),
            ShapeStyle {
                color: RGBAColor(100, 149, 237, 0.10),
                filled: false,
                stroke_width: 1,
            },
        ))?;
    }

    // Ensemble mean path — bold dark red
    let mean_path: Vec<(f64, f64)> = (0..=config.n_steps)
        .map(|i| {
            let m = paths.iter().map(|p| p[i]).sum::<f64>() / N_DISPLAY as f64;
            (i as f64 * dt, m)
        })
        .collect();

    chart
        .draw_series(LineSeries::new(
            mean_path,
            ShapeStyle {
                color: RGBAColor(200, 30, 30, 1.0),
                filled: false,
                stroke_width: 3,
            },
        ))?
        .label(format!("Ensemble mean  (n={})", N_DISPLAY))
        .legend(|(x, y)| {
            PathElement::new(
                vec![(x, y), (x + 25, y)],
                ShapeStyle {
                    color: RGBAColor(200, 30, 30, 1.0),
                    filled: false,
                    stroke_width: 3,
                },
            )
        });

    // Strike — orange horizontal reference line
    chart
        .draw_series(LineSeries::new(
            vec![(0.0, option.strike), (option.t, option.strike)],
            ShapeStyle {
                color: RGBAColor(255, 140, 0, 1.0),
                filled: false,
                stroke_width: 2,
            },
        ))?
        .label(format!("Strike  K = {:.0}", option.strike))
        .legend(|(x, y)| {
            PathElement::new(
                vec![(x, y), (x + 25, y)],
                ShapeStyle {
                    color: RGBAColor(255, 140, 0, 1.0),
                    filled: false,
                    stroke_width: 2,
                },
            )
        });

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.90))
        .border_style(BLACK)
        .position(SeriesLabelPosition::UpperLeft)
        .draw()?;

    root.present()?;
    Ok(())
}

/// Plot the running Monte Carlo price estimate as paths accumulate, showing
/// the ±1σ standard-error band narrowing and the estimate converging to
/// `final_price`. Saves to `output` (PNG).
pub fn plot_convergence(
    heston: &HestonParams,
    option: &OptionParams,
    config: &McConfig,
    seed: u64,
    final_price: f64,
    output: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(1337));
    let discount = (-option.r * option.t).exp();

    // Sample the running estimate every `stride` paths (≈800 points total).
    let stride = (config.n_paths / 800).max(1);
    // (n_paths_so_far, price, upper_1σ, lower_1σ)
    let mut pts: Vec<(f64, f64, f64, f64)> = Vec::new();

    let mut payoff_sum = 0.0_f64;
    let mut payoff_sq_sum = 0.0_f64;

    for i in 1..=config.n_paths {
        let s_t = simulate_path(heston, option, config, &mut rng);
        let p = if option.is_call {
            (s_t - option.strike).max(0.0)
        } else {
            (option.strike - s_t).max(0.0)
        };
        payoff_sum += p;
        payoff_sq_sum += p * p;

        if i % stride == 0 || i == config.n_paths {
            let n = i as f64;
            let mean = payoff_sum / n;
            let var = ((payoff_sq_sum / n) - mean * mean).max(0.0);
            let se = (var / n).sqrt() * discount;
            let price = mean * discount;
            // Clamp lower bound to zero — option prices cannot be negative.
            pts.push((n, price, price + se, (price - se).max(0.0)));
        }
    }

    // Y-axis: encompass all ±1σ bounds plus the final converged price.
    let y_lo = pts.iter().map(|e| e.3).fold(f64::INFINITY, f64::min);
    let y_hi = pts.iter().map(|e| e.2).fold(f64::NEG_INFINITY, f64::max);
    let pad = (y_hi - y_lo) * 0.15;
    let y_lo = (y_lo - pad).min(final_price * 0.75).max(0.0);
    let y_hi = (y_hi + pad).max(final_price * 1.25);

    let root = BitMapBackend::new(output, (1400, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let x_max = config.n_paths as f64;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!(
                "MC Convergence — Running Price Estimate   \
                 (converged = {:.4}  ±  {:.4})",
                final_price,
                pts.last().map(|e| e.2 - e.1).unwrap_or(0.0),
            ),
            ("sans-serif", 20).into_font(),
        )
        .margin(30)
        .x_label_area_size(50)
        .y_label_area_size(70)
        .build_cartesian_2d(0f64..x_max, y_lo..y_hi)?;

    chart
        .configure_mesh()
        .x_desc("Number of Paths")
        .y_desc("Estimated Option Price")
        .x_labels(10)
        .y_labels(10)
        .draw()?;

    // ±1σ confidence band as a filled polygon (upper boundary forward,
    // lower boundary reversed so the vertices form a closed ring).
    let band: Vec<(f64, f64)> = pts
        .iter()
        .map(|&(n, _, hi, _)| (n, hi))
        .chain(pts.iter().rev().map(|&(n, _, _, lo)| (n, lo)))
        .collect();

    chart.draw_series(std::iter::once(Polygon::new(
        band,
        ShapeStyle {
            color: RGBAColor(100, 149, 237, 0.20),
            filled: true,
            stroke_width: 0,
        },
    )))?;

    // Upper ±1σ bound
    chart
        .draw_series(LineSeries::new(
            pts.iter().map(|&(n, _, hi, _)| (n, hi)),
            ShapeStyle {
                color: RGBAColor(100, 149, 237, 0.55),
                filled: false,
                stroke_width: 1,
            },
        ))?
        .label("\u{00B1}1\u{03C3} std error")
        .legend(|(x, y)| {
            PathElement::new(
                vec![(x, y), (x + 25, y)],
                ShapeStyle {
                    color: RGBAColor(100, 149, 237, 0.80),
                    filled: false,
                    stroke_width: 2,
                },
            )
        });

    // Lower ±1σ bound
    chart.draw_series(LineSeries::new(
        pts.iter().map(|&(n, _, _, lo)| (n, lo)),
        ShapeStyle {
            color: RGBAColor(100, 149, 237, 0.55),
            filled: false,
            stroke_width: 1,
        },
    ))?;

    // Running mean price (main line)
    chart
        .draw_series(LineSeries::new(
            pts.iter().map(|&(n, price, _, _)| (n, price)),
            ShapeStyle {
                color: RGBAColor(30, 100, 220, 1.0),
                filled: false,
                stroke_width: 2,
            },
        ))?
        .label("Running estimate")
        .legend(|(x, y)| {
            PathElement::new(
                vec![(x, y), (x + 25, y)],
                ShapeStyle {
                    color: RGBAColor(30, 100, 220, 1.0),
                    filled: false,
                    stroke_width: 2,
                },
            )
        });

    // Final converged price — horizontal reference
    chart
        .draw_series(LineSeries::new(
            vec![(0.0, final_price), (x_max, final_price)],
            ShapeStyle {
                color: RGBAColor(200, 30, 30, 0.90),
                filled: false,
                stroke_width: 2,
            },
        ))?
        .label(format!("Converged  {:.4}", final_price))
        .legend(|(x, y)| {
            PathElement::new(
                vec![(x, y), (x + 25, y)],
                ShapeStyle {
                    color: RGBAColor(200, 30, 30, 0.90),
                    filled: false,
                    stroke_width: 2,
                },
            )
        });

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.90))
        .border_style(BLACK)
        .position(SeriesLabelPosition::UpperRight)
        .draw()?;

    root.present()?;
    Ok(())
}
