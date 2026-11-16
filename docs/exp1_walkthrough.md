# Experiment 1 walkthrough: artificial price series (paper Sec. IV-A)

Notes for the presentation. Written 2026-11-10. Details of every choice are in `NOTES.md`.

Regenerate everything:
```
python experiments/plot_synthetic_data.py --seed 0   # data figures (a few seconds)
python experiments/exp1_synthetic.py --seed 0        # trading results (about 1 minute)
pytest                                               # 23 tests
```
Status on 2026-11-10: all 23 tests pass, and rerunning `exp1_synthetic.py --seed 0`
gives exactly the same numbers as the saved `results/exp1_summary.json`.

---

## 1. The artificial data

Code: `src/data_synthetic.py`. Paper Eqs. 42–44.

```
beta(t) = 0.9 * beta(t-1) + nu(t)            nu  ~ N(0, 1)    hidden trend
p(t)    = p(t-1) + beta(t-1) + 3 * eps(t)    eps ~ N(0, 1)    log-price
z(t)    = exp(p(t) / R),  R = max(p) - min(p)                 price
```

- **Hidden trend `beta`**: an AR(1) process. A shock to it fades by 10% per step
  (half-life about 6.6 steps). Long-run standard deviation 1/sqrt(1 - 0.81) ≈ 2.3.
- **Log-price `p`**: each step moves by the trend (sd ≈ 2.3) plus noise (sd = 3).
  The noise is larger than the signal on every single step. This is the paper's
  "trending on short time scales with a high level of noise".
- **Price `z`**: dividing by the range R makes `p/R` span exactly 1, so prices always
  stay within a factor e (≈ 2.7) from lowest to highest. We start at `p(0) = 0`,
  so `z(0) = 1` (the paper doesn't give a start value).
- The trader sees `r_t = z_t - z_{t-1}`, **price differences, not log returns**.
- 10,000 samples per series: first 1,000 for training, last 9,000 for test.

### Why it is learnable
Autocorrelation of returns, measured over 100 series vs theory
(theory: `0.9^k * 5.26 / (5.26 + 9)`):

| Lag | 1 | 2 | 5 | 10 | 20 |
|---|---|---|---|---|---|
| Measured | 0.331 | 0.297 | 0.216 | 0.128 | 0.044 |
| Theory | 0.332 | 0.299 | 0.218 | 0.129 | 0.045 |

The close match checks that the generator is correct. Predictability is real but short-lived:
most of it is in the last ~8 returns, which is exactly what the trader is given (M = 7, 8 lags).

### Why costs matter so much
The median one-step price move |r_t| is **0.0016** (the mean is 0.0022, because a few moves are large).
A long→short reversal costs 2δ:

| Cost | δ | 2δ | × median move |
|---|---|---|---|
| 0.2% | 0.002 | 0.004 | ~3 |
| 0.5% | 0.005 | 0.010 | ~6 |
| 1.0% | 0.010 | 0.020 | ~13 |

At 1% cost, one reversal costs about 13 typical price moves. The trader can only profit by
holding positions through long trends, so it should trade less as cost rises.

### Data figures
`results/synthetic_data_overview.png`: one full series with the train/test split, a 300-step
zoom, and the hidden trend β(t) over the same window (green = up-trend, orange = down-trend).
The trader never sees β; it has to infer it from past prices.

![overview](../results/synthetic_data_overview.png)

`results/synthetic_data_properties.png`: three independent series (each trial gets a fresh
one), autocorrelation vs theory, and the distribution of |r_t| with the reversal costs marked.

![properties](../results/synthetic_data_properties.png)

Suggested slide order: "what the data looks like" (overview) → "why it is learnable but costly
to trade" (properties) → results.

---

## 2. Did we match the paper's numbers?

**No, not exactly, and an exact match was never possible.** Every trial uses freshly generated
random prices, and the paper does not publish its random seeds, learning rate, initial weights
or input scaling. The realistic target is the same qualitative behaviour with numbers in the
same range.

Test-set medians over 100 trials (paper values read off its Fig. 5):

| Cost | Sharpe (ours / paper) | Profit % (ours / paper) | Trade freq % (ours / paper) | Trials with Sharpe > 0 (ours / paper) |
|---|---|---|---|---|
| 0.2% | 0.184 / 0.213 | 550 / 590 | 7.2 / 6.0 | 100% / 100% |
| 0.5% | 0.091 / 0.137 | 340 / 430 | 4.6 / 4.2 | 98% / 100% |
| 1.0% | 0.025 / 0.059 | 114 / 225 | 3.2 / 3.1 | 70% / 100% |

Sharpe is per period, not annualized (this matches the scale of the paper's Fig. 5).

### What we reproduce
- The trader learns to make money on this data.
- Sharpe and profit fall as cost rises.
- Trading frequency drops with cost, almost exactly as in the paper.
- The single from-scratch run (Fig. 2) loses money while learning, then profits steadily:
  about +130% over 10,000 periods, vs about +200% in the paper's run.
  We show seed 0 rather than picking a good-looking seed.

### What we don't reproduce
- Our Sharpe and profit are lower, and the gap grows with cost. At 1% our median profit is
  half the paper's.
- The paper reports all 100 trials positive; we get 100% / 98% / 70% at 0.2% / 0.5% / 1%.
- Our last-1,000-period window (Fig. 3) includes a losing stretch, unlike the paper's.

### Where we think the gap comes from
Not the signal. With zero cost our trader reaches Sharpe 0.34, the same as a simple
least-squares predictor fitted on the same 8 lags (0.341), so it extracts all the predictability
there is. With costs it falls behind both that benchmark (0.200 / 0.107 / 0.063) and the paper.
So the weakness is in **when it chooses to trade given the costs**. The paper doesn't give
enough detail to know what it did differently there.

Result figures: `results/exp1_fig2_single_run.png`, `results/exp1_fig3_last1000.png`,
`results/exp1_fig5_boxplots.png`.

---

## 3. Did we change the paper's formulas?

**No.** All of these follow the paper exactly:
- the price generator (Eqs. 42–44)
- the trading return `R_t = mu * (F_{t-1} r_t - delta |F_t - F_{t-1}|)`, with mu = 1
- the differential Sharpe ratio and its derivative, with A_{t-1}, B_{t-1} treated as constants, η = 0.01
- the recurrent gradient, including the term through the previous position `dF_{t-1}/dθ`

The tests check the hand-derived gradient against finite differences and PyTorch autograd
(tolerance 1e-5, float64). They also check that dropping the recurrent term gives a wrong
gradient, that there is no look-ahead, and that results are deterministic.

The differences fall into three groups.

### 3a. Where we followed the paper instead of CLAUDE.md
`CLAUDE.md` describes a continuous `tanh` position. The paper says the Exp. 1 traders are
"threshold units" that are always fully long or fully short (+1 / −1), and its Fig. 2 signal
is ±1. So we trade `F_t = sign(tanh(...))` and take the gradient through the smooth `tanh`
value before thresholding (the paper's "Adaline"-style rule, a straight-through estimator).
**This is the main implementation decision.** `rrl.py` supports both modes, and both are
gradient-tested.

### 3b. Choices we had to make (the paper is silent)
| Setting | Our choice | How chosen |
|---|---|---|
| Learning rate ρ (100-trial runs) | 0.001 | pilot seeds 1000–1015, never the reported trials 0–99 |
| Learning rate ρ (single from-scratch run) | 0.01 | same pilot seeds |
| Initial weights | N(0, 0.1²) | 0.01 and 0.1 gave the same results; 1.0 was worse |
| Input scaling | divide by the std of returns in the training window | training data only |
| DSR start-up | first 100 steps (= 1/η) update A, B but not the weights | avoids dividing by ~0 |
| Cost units | "0.2%" means δ = 0.002 per unit change in position | prices start at 1 |
| Epoch | one online pass over the 1,000 training samples | paper's "real-time recurrent learning" |
| First decision | at t = 8, starting flat; the first trade pays the cost | we count it in the statistics |
| Trials | trial i uses seed i at every cost level (same 100 series) | paper only says "different realizations" |

### 3c. A variant we tried and rejected
A different reading of the "Adaline" rule for the cost term: evaluate `dR/dF` at the smooth
pre-threshold value. On pilot seeds it traded far too rarely (1.4% vs the paper's ~4.2% at 0.5%
cost) and was no better, so we kept the straight-through version.

---

## 4. One-line summary for the slides
"Qualitatively reproduced; our Sharpe is lower at higher costs; the formulas are the paper's;
the unspecified settings are documented."

## 5. Possible next step
Test whether the cost gap closes when we vary the unspecified settings that affect trade
timing (e.g. DSR warm-up, learning rate during the test period). Any such test runs on fresh
pilot seeds, never on the reported trials, and gets logged in `NOTES.md`.
