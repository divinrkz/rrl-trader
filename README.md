# Recurrent RL Trader

Implementation of Moody & Saffell, *Learning to Trade via Direct Reinforcement*, IEEE Trans. Neural Networks 12(4), 2001.

## Method

Basically, a trader is a single recurrent tanh unit,

$$F_t = \mathrm{sign}(\tanh(\theta \cdot [1, x_t, F_{t-1}]))$$

that takes a long (+1) or short (−1) positions. Its trading return is

$$R_t = F_{t-1} r_t - \delta \lvert F_t - F_{t-1} \rvert$$

which is just the profit from the position held over the period, minus the cost of changing it. Instead of forecasting prices or learning a value function, **Recurrent Reinforcement Learning (RRL)** adjusts $\theta$ by gradient ascent directly on a trading performance measure. We use both the **differential Sharpe ratio** $D_t$, the first-order effect of the latest return on an exponential moving Sharpe ratio, and **differential downside deviation** which allows on-line updates. The paper also compares this direct reinforcement approach with Q-learning on monthly S&P 500 / T-bill allocation.

Here is the [paper](https://ieeexplore.ieee.org/document/935097).

### Run

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest                                 
.venv/bin/python experiments/exp1_synthetic.py --seed 0   
.venv/bin/python experiments/fetch_data.py                
.venv/bin/python experiments/exp2_sp500.py --seed 0        
.venv/bin/jupyter lab notebooks/results.ipynb   
```
