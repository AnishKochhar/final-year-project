import numpy as np, matplotlib.pyplot as plt
from pathlib import Path

def plot_fc_grid(fc_list, *, save_path: Path, vmax=0.8):
    """ Draws a 4x4 grid of FC matrices (expects 16 mats of shape NxN) """
    fig, axs = plt.subplots(4, 4, figsize=(8,8))
    for i, (ax, fc) in enumerate(zip(axs.flat, fc_list)):
        im = ax.imshow(fc, vmin=-vmax, vmax=vmax, cmap="coolwarm")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"W{i}", fontsize=6)
    plt.tight_layout()
    plt.colorbar(im, ax=axs.ravel().tolist(), shrink=0.5)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
