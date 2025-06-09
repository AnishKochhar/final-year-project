import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from whobpyt.data_loader import BOLDDataLoader, DEVICE

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/vol/bitbucket/ank121/fyp/HCP Data")
    p.add_argument("--subj-id", type=int, default=18)
    p.add_argument("--save-dir", type=str, default="plots/report_figures")

    args = p.parse_args()

    fmri = os.path.join(args.data_root, "BOLD Timeseries HCP.mat")
    scdir = os.path.join(args.data_root, "distance_matrices")
    dist  = os.path.join(args.data_root, "schaefer100_dist.npy")
    loader = BOLDDataLoader(fmri, scdir, dist, chunk_length=100)
    loader._split_into_chunks()

    # Plot original SC
    sc_orig = loader.all_SC[args.subj_id]
    vabs = abs(sc_orig).max()
    vmin = -vabs if sc_orig.min() < 0 else 0.0
    cmap = "coolwarm" if vmin < 0 else "viridis"

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(sc_orig, 
                vmin=vmin, 
                vmax=vabs, 
                cmap=cmap, 
                square=True, 
                ax=ax, 
                cbar_kws={"label": "Connection Strength"},
                xticklabels=False,
                yticklabels=False)
    
    # ax.set_title("Original Structural Connectivity", pad=20, fontsize=12)
    if args.subj_id is not None:
        fig.suptitle(f"Subject {args.subj_id}", y=0.95, fontsize=14)
    
    plt.tight_layout()
    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, "sc_orig.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved {path}")

    # Plot normalized SC
    sc_norm = loader.get_subject_connectome(args.subj_id, norm=True).cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(sc_norm,
                vmin=0,
                vmax=1,
                cmap="viridis",
                square=True,
                ax=ax,
                cbar_kws={"label": "Normalized Connection Strength"},
                xticklabels=False,
                yticklabels=False)
    
    # ax.set_title("Normalized Structural Connectivity", pad=20, fontsize=12)
    if args.subj_id is not None:
        fig.suptitle(f"Subject {args.subj_id}", y=0.95, fontsize=14)
    
    plt.tight_layout()
    path = os.path.join(args.save_dir, "sc_norm.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved {path}")

if __name__ == "__main__":
    main()



