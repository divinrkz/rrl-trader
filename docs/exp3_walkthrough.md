# Experiment 3 walkthrough: USD/GBP FX trader (paper Sec. IV-B)

Notes for later reading. Written 2026-11-11. Details of every choice are in `NOTES.md`.

> **Naming:** this is the **second** experiment in the paper (Sec. IV-B). In this project it is called
> "Experiment 3", because we built the S&P experiment (paper Sec. IV-C) first and called it "Experiment 2".

> **SUBSTITUTE DATA.** The paper used Olsen & Associates HFDF96 half-hourly USD/GBP quotes for
> Jan–Aug 1996. They are commercial. We use **HistData.com GBP/USD tick data (bid and ask) for
> Jan–Aug 2010**. **These are not results on the paper's data.**

Regenerate everything (about 2 minutes on 8 cores):
```
python experiments/exp3_pilot.py --seed 100     # choose settings on Sep-Dec 2009 only
python experiments/exp3_fx.py --seed 0          # the experiment: 25 traders on Jan-Aug 2010
python experiments/exp3_diagnose.py --seed 0    # where the losses come from (+ exp3_why_it_fails.png)
```
The data must be in `data/raw/histdata/` as the 12 monthly zip files (`HISTDATA_COM_ASCII_GBPUSD_T200909.zip` … `T201008.zip`).

---

## 1. What the paper does

- **Market:** the US dollar / British pound exchange rate, quoted every half hour, 24 hours a day,
  5 days a week, for the first 8 months of 1996.
- **Three positions:** long the pound, short the pound, or **neutral** (out of the market).
  Exps. 1 and 2 only had long/short.
- **Real trading costs:** the data has bid and ask prices, and the trader must buy at the ask and
  sell at the bid. The cost is the spread itself, not an assumed percentage.
- **Objective:** the **differential downside deviation ratio** (DDR), Eqs. 21–24. It is like the Sharpe
  ratio, but only losses count as risk:
  ```
  DD  = sqrt( mean( min(R_t, 0)² ) )        downside deviation
  DDR = mean(R) / DD
  ```
  The differential version updates moving averages online, as the differential Sharpe ratio does:
  ```
  A_t   = A_{t-1}   + η (R_t − A_{t-1})
  DD²_t = DD²_{t-1} + η (min(R_t,0)² − DD²_{t-1})
  D_t = (R_t − ½A_{t-1}) / DD_{t-1}                                         if R_t > 0
  D_t = (DD²_{t-1}(R_t − ½A_{t-1}) − ½A_{t-1}R_t²) / DD³_{t-1}            if R_t ≤ 0
  ```
  A gain is never penalized, however large. Losses are penalized quadratically.
- **Walk-forward protocol:** train on the first 2,000 half-hour bars (about 2 months), trade the next 480
  (2 weeks) with the trained weights, shift everything forward by 2 weeks, retrain, and repeat.
- **Paper's result (one trader, Fig. 6):** about **15% a year**, **Sharpe 2.3**, a trade **every 5 hours**.
  The paper warns these are indicative quotes and might not be tradeable.

## 2. Our data

| | Paper | Ours |
|---|---|---|
| Source | Olsen HFDF96 (commercial) | HistData.com free tick data |
| Period | Jan–Aug 1996 | Jan–Aug 2010 (pilot: Sep–Dec 2009) |
| Bid and ask | yes | yes |
| Spread | not reported | 3–5 pips, then 7 pips in Jul–Aug 2010 |

**How the half-hour series is built** (`src/data_fx.py`):
- Read every tick (time, bid, ask). HistData times are EST without daylight saving, so add 5 hours to get UTC.
- For each 30-minute bar, keep the **last** bid and ask (the quote at the bar's close). Drop bars with no
  ticks (weekends, holidays). Drop the 5 ticks where bid ≥ ask.
- `mid = (bid + ask)/2`, return `r_t = mid_t/mid_{t−1} − 1`, and cost rate `δ_t = (ask − bid)/(2·mid)`, half the spread.
- Trading return: `R_t = F_{t−1}·r_t − δ_t·|F_t − F_{t−1}|`. Going neutral → long pays half the spread,
  long → short pays the full spread. A test checks that this equals buying at the ask and selling at the bid.

Test data: 8,278 bars, the first 2,000 for initial training, then **6,278 out-of-sample bars**
(2 March – 31 August 2010, about 6 months, like the paper's).

**Why not other sources:** Dukascopy 2004 was tried first, but the server throttled the download to
5–6 days. TrueFX now only offers 2025–26, and FXCM's free archive is behind a login.

## 3. The trader and choices the paper doesn't give

- `y_t = tanh(w · [1, last 8 returns, F_{t−1}])`, and the position is **long if y > ½, short if y < −½,
  neutral otherwise**. The paper doesn't give the thresholds.
- Inputs: the last 8 half-hour returns (as in Exp. 1), scaled by the training window's std.
- Each 2-week retraining starts from the previous weights and makes `epochs` online passes over the 2,000 bars.
- Learning rate ρ and number of epochs: chosen on the **pilot months (Sep–Dec 2009)** only, with a rule fixed
  before running: highest mean Sharpe over pilot seeds 100–107. Result: **ρ = 0.03, 30 epochs**.
  **Every pilot setting lost money** (Sharpe −1.8 to −4.1), so the pilot didn't really separate the settings.
- 25 traders (seeds 0–24). The paper shows one. We report the median and spread.
- Annualization: about 12,500 half-hour bars a year (48 a day × 5 days × 52 weeks).

## 4. Results (Jan–Aug 2010, out of sample)

| | Ours (median of 25) | Paper (one trader) |
|---|---|---|
| Annualized return | **−55%** | +15% |
| Annualized Sharpe | **−4.5** | 2.3 |
| Annualized DDR | −6.1 | — |
| Hours between trades | 3.6 | ≈ 5 |
| Traders with positive Sharpe | **0 / 25** | — |
| Time short / long / neutral | 71% / 27% / 1% | — |
| Baseline: hold GBP long | +6.2%, Sharpe 0.51 | — |

**The paper's FX result is not reproduced.** All 25 traders lose money, steadily, over the whole period.
(Returns are additive with position size 1, so −55% a year is a sum of returns, not compounded wealth.)

![single run](../results/exp3_fx/exp3_fig6_single_run.png)

`results/exp3_fx/exp3_fig6_single_run.png`: seed 0, in the layout of the paper's Fig. 6. Panels: price,
position (switching every few hours), cumulative return (a steady decline) and the moving Sharpe ratio (mostly below 0).

![distribution](../results/exp3_fx/exp3_distribution.png)

`results/exp3_fx/exp3_distribution.png`: all 25 traders, with the paper's numbers as dashed lines.
Trading frequency is close to the paper's, but returns and Sharpe are far below it.

## 5. Why it fails: two separate problems

`experiments/exp3_diagnose.py` splits each trader's return into **gross** profit (`F_{t−1}·r_t`, what the
positions earned) and **spread cost**. It also re-runs the traders with zero cost. That zero-cost run is a
diagnostic, not a result. Nothing was re-tuned. Output: `results/exp3_fx/exp3_cost_decomposition.txt`.

| (annualized, median of 25) | |
|---|---|
| Gross profit from positions | −3.9% (11 of 25 positive) |
| Spread cost paid | 52.6% |
| Net | −54.8% |
| Zero-cost Sharpe (train and test without cost) | 0.07 (14 of 25 positive) |

![why it fails](../results/exp3_fx/exp3_why_it_fails.png)

`results/exp3_fx/exp3_why_it_fails.png` **(slide figure)**. Left: each trader's annualized return split into
profit from its positions, spread paid, and the net result; bars are medians, dots are the 25 traders.
Right: the same traders run with zero trading cost. They scatter around 0 (median 0.07), so there is no
reliable signal even before costs. A few seeds look good by chance.

1. **There was no exploitable signal in 2010.** Even without any cost, the traders are at about zero:
   half win, half lose. In the 2009 pilot months, half-hour returns had a small reversal pattern
   (lag-2 autocorrelation −0.037), and the traders found it (zero-cost Sharpe ≈ 3.6 in the pilot). It did not
   carry over into 2010.
2. **The traders trade far too much for the spread.** A trade every 3.6 hours, often a full long → short
   reversal, costs about 53% a year. The spread also widens in Jun–Aug 2010 (7 pips; median half-spread
   1.3 → 2.3 basis points against typical half-hour moves of 10.7 bp). The DDR objective in principle
   penalizes costs, but with no real signal the trader keeps reacting to noise.

**What we can't tell:** the paper doesn't report the Olsen spreads or the trader's inputs, so we can't say
whether the gap is from the data (2010 vs 1996, different spreads) or from details of the method.

## 6. Summary for the slides (if used)

- The FX experiment is the hardest to reproduce: the original data is commercial, and the paper gives few
  details (inputs, thresholds, learning rate).
- On free 2010 bid/ask data, the RRL trader **does not** reproduce the paper's 15% / Sharpe 2.3. It loses
  steadily because (1) there was no stable short-term pattern to exploit and (2) it traded too often for the spread.
- **Lesson:** at high frequency, transaction costs are as large as the signal. An effect visible in one
  2-month window (the pilot) can disappear in the next.

## 7. Possible follow-ups (new experiments, not tuning on 2010)
- A wider neutral zone (e.g. long only if y > 0.8), or a cost multiplier during training, to force fewer trades.
- Richer inputs (time of day, volatility, other currency pairs).
- Each must be chosen on pilot data and reported as a new experiment.
