import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# Load data
SAVE_DIR = Path("exp2_outputs")
res_mean = np.load(SAVE_DIR / "grid_fc_corr.npy")
with open(SAVE_DIR / "grid_axes.json") as f:
    axes = json.load(f)

g_vals = np.array(axes["g"])
g_EE_vals = np.array(axes["g_EE"])

# Slice to remove g < 30
mask = (g_vals >= 30) & (g_vals <= 130)
res_mean_trimmed = res_mean[:, mask]
g_vals_trimmed = g_vals[mask]
# Find the indices of the maximum value in the trimmed result
max_idx = np.unravel_index(np.argmax(res_mean_trimmed), res_mean_trimmed.shape)
max_g = g_vals_trimmed[max_idx[1]]
max_gEE = g_EE_vals[max_idx[0]]
max_val = res_mean_trimmed[max_idx]
print(f"Max FC-corr: {max_val:.8f} at g={max_g}, g_EE={max_gEE} (indices: {max_idx})")

# Compute log-scaled distance to optimum
delta_r = max_val - res_mean_trimmed
# To avoid log(0), add a small epsilon
epsilon = 1e-8
log_delta_r = np.log10(delta_r + epsilon)

plt.figure(figsize=(6, 4))
sns.heatmap(log_delta_r, cmap="magma",
            xticklabels=g_vals_trimmed, yticklabels=g_EE_vals,
            cbar_kws={"label": r"$\log_{10}(\Delta r_{\mathrm{FC}})$"})

# Optional: mark ground truth (if still within range)
if 80 in g_vals_trimmed and 3.5 in g_EE_vals:
    x = list(g_vals_trimmed).index(80) + 0.5
    y = list(g_EE_vals).index(3.5) + 0.5
    plt.scatter(x, y, c="cyan", marker="x", s=60)

plt.xlabel("g")
plt.ylabel("g_EE")
plt.title(r"Log-scaled distance to optimum: $\Delta r_{\mathrm{FC}} = r_{\max} - r_{\mathrm{FC}}$")
plt.tight_layout()
plt.savefig(SAVE_DIR / "heatmap_log_delta_r.png", dpi=300)
plt.close()


# # Plot
# plt.figure(figsize=(6, 4))
# sns.heatmap(res_mean_trimmed, cmap="viridis",
#             xticklabels=g_vals_trimmed, yticklabels=g_EE_vals,
#             cbar_kws={"label": "mean FC-corr"})

# # Optional: mark ground truth (if still within range)
# if 80 in g_vals_trimmed and 3.5 in g_EE_vals:
#     x = list(g_vals_trimmed).index(80) + 0.5
#     y = list(g_EE_vals).index(3.5) + 0.5
#     plt.scatter(x, y, c="red", marker="x", s=60)

# plt.xlabel("g"); plt.ylabel("g_EE")
# plt.title("FC-corr heatmap")
# plt.tight_layout()
# plt.savefig(SAVE_DIR / "heatmap_grid_trimmed.png", dpi=300)
# plt.close()
