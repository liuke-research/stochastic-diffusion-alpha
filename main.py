from data.market_state import build_market_state
from data.market_loader import MarketLoader

import numpy as np
import matplotlib.pyplot as plt


from spde.simulator import run_simulation
from diffusion.feature import FeatureBuilder

from alpha.evaluation import ic, rank_ic
from alpha.backtest import Backtester

from analysis.mixing import mixing_curve
from analysis.large_deviation import dv_rate



# =========================
# 1. Load market data
# =========================

loader = MarketLoader()

market = loader.load()

print(market.head())


returns = market["return"].values



# =========================
# 2. Build rolling states
# =========================

print("Building market states...")


window = 100


states = build_market_state(
    returns,
    window=window
)



# =========================
# 3. Generate SPDE Alpha Series
# =========================

print("Generating SPDE alpha...")


fb = FeatureBuilder()


alpha_series = []


for i, state in enumerate(states):

    if i % 500 == 0:
        print(
            f"Processing state {i}/{len(states)}"
        )


    traj = run_simulation(
        initial_state=state,
        T=80,
        dim=100
    )


    x0 = traj[0]

    xT = traj[-1]


    alpha = fb.build_signal(
        x0,
        xT
    )


    alpha_series.append(alpha)



alpha_series = np.array(alpha_series)



print(
    "Alpha generation finished."
)



# =========================
# 4. Future return alignment
# =========================

# 预测未来5日收益

horizon = 5


future_returns = []


for i in range(
    window,
    len(returns)-horizon
):

    future = np.sum(
        returns[i:i+horizon]
    )

    future_returns.append(
        future
    )


future_returns = np.array(
    future_returns
)



# 保证长度一致

min_len = min(
    len(alpha_series),
    len(future_returns)
)


alpha_series = alpha_series[:min_len]

future_returns = future_returns[:min_len]



# =========================
# 5. IC evaluation
# =========================

print(
    "Computing IC..."
)


ic_value = ic(
    alpha_series,
    future_returns
)


rank_ic_value = rank_ic(
    alpha_series,
    future_returns
)



print(
    f"IC: {ic_value:.4f}"
)


print(
    f"Rank IC: {rank_ic_value:.4f}"
)



# =========================
# 6. Backtest
# =========================

print(
    "Running backtest..."
)


bt = Backtester()


equity = bt.run(
    alpha_series,
    future_returns
)



plt.figure()

plt.plot(
    equity
)

plt.title(
    "SPDE Alpha Equity Curve"
)

plt.savefig(
    "plots/equity_curve.png"
)



# =========================
# 7. Alpha distribution
# =========================

plt.figure()

plt.hist(
    alpha_series,
    bins=50
)

plt.title(
    "SPDE Alpha Distribution"
)


plt.savefig(
    "plots/alpha_distribution.png"
)



# =========================
# 8. Mixing analysis
# =========================

print(
    "Analyzing mixing..."
)


mix = mixing_curve(
    states[-1]
)



plt.figure()

plt.plot(
    mix
)

plt.title(
    "Mixing Decay"
)


plt.savefig(
    "plots/mixing.png"
)



# =========================
# 9. DV rate
# =========================

print(
    "Estimating DV rate..."
)


dv = dv_rate(
    states[-1]
)



print(
    f"DV rate: {dv:.4f}"
)



# =========================
# 10. Summary
# =========================


print(
    "\n===== SUMMARY ====="
)


print(
    f"IC        : {ic_value:.4f}"
)


print(
    f"Rank IC   : {rank_ic_value:.4f}"
)


print(
    f"DV rate   : {dv:.4f}"
)


print(
    "==================="
)


print(
    "\nAll results saved in /plots"
)