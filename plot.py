import matplotlib.pyplot as plt
import numpy as np

# Resource labels and totals
resources = ["CPUs", "Memory"]
totals = [9, 18]  # total per resource

# User allocations (same units as totals)
userA = [3, 12]
userB = [6, 2]

# Normalize to percentages
userA_pct = np.array(userA) / np.array(totals) * 100
userB_pct = np.array(userB) / np.array(totals) * 100

# Plot
fig, ax = plt.subplots(figsize=(4, 4))
x = np.arange(len(resources))

ax.bar(x, userB_pct, color="#264b7b", label="User B")
ax.bar(x, userA_pct, bottom=userB_pct, color="#a6c4e0", label="User A")

# Annotate
ax.text(
    0,
    userB_pct[0] / 2,
    f"{userB[0]} CPUs",
    color="white",
    ha="center",
    va="center",
    fontsize=10,
    fontweight="bold",
)
ax.text(
    0,
    userB_pct[0] + userA_pct[0] / 2,
    f"{userA[0]} CPUs",
    color="black",
    ha="center",
    va="center",
    fontsize=10,
)
ax.text(
    1,
    userB_pct[1] / 2,
    f"{userB[1]} GB",
    color="white",
    ha="center",
    va="center",
    fontsize=10,
    fontweight="bold",
)
ax.text(
    1,
    userB_pct[1] + userA_pct[1] / 2,
    f"{userA[1]} GB",
    color="black",
    ha="center",
    va="center",
    fontsize=10,
)

# Aesthetics
ax.set_xticks(x)
ax.set_xticklabels(
    [f"{r}\n({t} total)" for r, t in zip(resources, totals)], fontsize=10
)
ax.set_ylabel("Utilization (%)", fontsize=11)
ax.set_ylim(0, 100)
ax.axhline(50, linestyle="--", color="black", linewidth=1)
ax.legend(frameon=False, loc="upper center", ncol=2)
plt.tight_layout()
plt.show()
