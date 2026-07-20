import numpy as np
import matplotlib.pyplot as plt

from spde.simulator import run_simulation
from diffusion.feature import FeatureBuilder
from alpha.evaluation import ic, rank_ic
from alpha.backtest import Backtester
from analysis.mixing import mixing_curve
from analysis.large_deviation import dv_rate


# =========================
# 1. SPDE simulation
# =========================
print("Running SPDE simulation...")

traj = run_simulation(T=200, dim=100)

x0 = traj[0]
xT = traj[-1]

plt.figure()
plt.plot(x0[:50], label="Initial")
plt.plot(xT[:50], label="Final")
plt.legend()
plt.title("SPDE State Evolution")
plt.savefig("plots/spde_evolution.png")


# =========================
# 2. Diffusion feature
# =========================
print("Building diffusion signal...")

fb = FeatureBuilder()
signal = fb.build_signal(x0, xT)


# =========================
# 3. Synthetic returns (demo)
# =========================
returns = np.random.randn(len(signal))


# =========================
# 4. Alpha evaluation
# =========================
print("Computing IC...")

ic_value = ic(signal, returns)
rank_ic_value = rank_ic(signal, returns)

print(f"IC: {ic_value:.4f}")
print(f"Rank IC: {rank_ic_value:.4f}")


# =========================
# 5. Backtest
# =========================
bt = Backtester()
equity = bt.run(signal, returns)

plt.figure()
plt.plot(equity)
plt.title("Equity Curve")
plt.savefig("plots/equity_curve.png")


# =========================
# 6. Mixing analysis
# =========================
print("Analyzing mixing...")

mix = mixing_curve(xT[:100])

plt.figure()
plt.plot(mix)
plt.title("Mixing Decay")
plt.savefig("plots/mixing.png")


# =========================
# 7. DV large deviation
# =========================
print("Estimating DV rate...")

dv = dv_rate(xT)

print(f"DV rate: {dv:.4f}")


# =========================
# 8. Summary
# =========================
print("\n===== SUMMARY =====")
print(f"IC        : {ic_value:.4f}")
print(f"Rank IC   : {rank_ic_value:.4f}")
print(f"DV rate   : {dv:.4f}")
print("===================")


print("\nAll results saved in /plots")