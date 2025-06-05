import numpy as np
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from whobpyt.data_loader import BOLDDataLoader

# Config
DATA_ROOT = "/vol/bitbucket/ank121/fyp/HCP Data"
SUBJECT_ID = 22  # Choose one subject
CHUNK_SIZES = [20, 50, 100]  # Short chunks for training
FULL_TRS = 1189  # Entire session for evaluation

def compute_fc(ts):
    return np.corrcoef(ts)

def main():
    fmri = f"{DATA_ROOT}/BOLD Timeseries HCP.mat"
    scdir = f"{DATA_ROOT}/distance_matrices"
    dist  = f"{DATA_ROOT}/schaefer100_dist.npy"
    outpath = Path("plots/debug")
    outpath.mkdir(parents=True, exist_ok=True)
    
    loader = BOLDDataLoader(fmri, scdir, dist)
    loader._split_into_chunks()

    ts_full = loader.all_bold[SUBJECT_ID]
    region_count = ts_full.shape[0]

    fcs = {}
    for chunk in CHUNK_SIZES:
        ts_chunk = ts_full[:, :chunk]
        fcs[f"{chunk} TR"] = compute_fc(ts_chunk)

    # Full-length FC
    fcs[f"{FULL_TRS} TR (Full)"] = compute_fc(ts_full[:, :FULL_TRS])

    # Plotting
    fig, axes = plt.subplots(1, len(fcs), figsize=(16, 4))
    for ax, (label, fc) in zip(axes, fcs.items()):
        im = ax.imshow(fc, vmin=-1, vmax=1, cmap='coolwarm')
        ax.set_title(label)
        ax.axis('off')

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
    plt.suptitle(f"Effect of Chunk Length on FC (Subject {SUBJECT_ID})", fontsize=14)
    plt.tight_layout()
    plt.savefig(outpath / "chunksize.png")

if __name__ == "__main__":
    main()
