""" Compare empirical vs simulated FC over multiple chunk sizes """

import os, argparse, numpy as np, torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from whobpyt.data_loader import BOLDDataLoader, DEVICE
from simulators.rww_simulator import RWWSubjectSimulator
from whobpyt.custom_cost_RWW import CostsRWW
from whobpyt.modelfitting import Model_fitting


def corrcoef_lower(a, b):
    """ Pearson r of lower-triangles of two square matrices """
    idx = np.tril_indices_from(a, k=-1)
    return np.corrcoef(a[idx], b[idx])[0, 1]

def fc(mat):
    return np.corrcoef(mat)

def plot_comparison(ts_sim, ts_emp, full_length, subj=18, title=""):
    CHUNK_SIZES = [50, 100, 200]
    labels = CHUNK_SIZES + [full_length]

    emp_FC_dict, sim_FC_dict, corr_dict = {}, {}, {}

    for size in labels:
        emp_FC_dict[size] = fc(ts_emp[:, -size:])
        sim_FC_dict[size] = fc(ts_sim[:, -size:])
        corr_dict[size] = corrcoef_lower(emp_FC_dict[size], sim_FC_dict[size])

    n_cols = len(labels)
    fig, axs = plt.subplots(2, n_cols, figsize=(4*n_cols, 8))

    for col, lbl in enumerate(labels):
        vmax = 1
        sns.heatmap(emp_FC_dict[lbl], vmin=-vmax, vmax=vmax, cmap="coolwarm", ax=axs[0, col], cbar=False, square=True)
        sns.heatmap(sim_FC_dict[lbl], vmin=-vmax, vmax=vmax, cmap="coolwarm", ax=axs[1, col], cbar=False, square=True)
        axs[0, col].set_title(f"Emp {lbl}")
        axs[1, col].set_title(f"Sim {lbl}\nr={corr_dict[lbl]}")
        axs[0, col].axis('off'); axs[1, col].axis('off')

    fig.suptitle(f"FC comparison @ chunk sizes (Subject {subj})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out_dir = Path("plots/debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"emp_vs_sim_FC_subj{subj}_{title}.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Plot] saved → {save_path}")


    print("\nLower-tri FC correlations (emp vs sim) [NO TRAIN]:")
    for k in CHUNK_SIZES:
        print(f"  {k:4d} TR : r = {corr_dict[k]:.3f}")
    print(f"  full ({full_length} TR) : r = {corr_dict[full_length]:.3f}")

def plot_chunkwise_fc(ts_sim, ts_emp, subj=18, title=""):
    """ For each non-overlapping 200-TR window, plot empirical and simulated FC matrices """
    chunk_length = 200
    n_roi, n_trs = ts_emp.shape
    n_chunks = n_trs // chunk_length

    emp_FC_list = []
    sim_FC_list = []
    corr_list = []

    for i in range(n_chunks):
        start = i * chunk_length
        end = start + chunk_length
        emp_chunk = ts_emp[:, start:end]
        sim_chunk = ts_sim[:, start:end]
        emp_fc = fc(emp_chunk)
        sim_fc = fc(sim_chunk)
        emp_FC_list.append(emp_fc)
        sim_FC_list.append(sim_fc)
        corr = corrcoef_lower(emp_fc, sim_fc)
        corr_list.append(corr)

    fig, axs = plt.subplots(2, n_chunks, figsize=(4*n_chunks, 8))
    if n_chunks == 1:
        axs = np.array(axs).reshape(2, 1)

    vmax = 1
    for i in range(n_chunks):
        sns.heatmap(emp_FC_list[i], vmin=-vmax, vmax=vmax, cmap="coolwarm", ax=axs[0, i], cbar=False, square=True)
        sns.heatmap(sim_FC_list[i], vmin=-vmax, vmax=vmax, cmap="coolwarm", ax=axs[1, i], cbar=False, square=True)
        axs[0, i].set_title(f"Emp chunk {i+1}")
        axs[1, i].set_title(f"Sim chunk {i+1}\nr={corr_list[i]:.3f}")
        axs[0, i].axis('off'); axs[1, i].axis('off')

    fig.suptitle(f"FC comparison per 200TR chunk (Subject {subj})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out_dir = Path("plots/debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"chunkwise_subj{subj}_{title}.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Plot] saved → {save_path}")

    print("\nChunkwise lower-tri FC correlations (emp vs sim):")
    for i, r in enumerate(corr_list):
        print(f"  Chunk {i+1:2d} ({chunk_length} TR): r = {r:.3f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/vol/bitbucket/ank121/fyp/HCP Data")
    ap.add_argument("--subj-id", type=int, default=18)
    ap.add_argument("--norm", action="store_true")
    ap.add_argument("--g", type=float, default=1000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--lambda-rate", type=float, default=0.05)
    ap.add_argument("--lambda-spec", type=float, default=0.05)
    ap.add_argument("--lambda-disp", type=float, default=0.03)
    args = ap.parse_args()

    train_chunk_length = 100

    fmri = os.path.join(args.data_root, "BOLD Timeseries HCP.mat")
    scdir = os.path.join(args.data_root, "distance_matrices")
    dist  = os.path.join(args.data_root, "schaefer100_dist.npy")
    loader = BOLDDataLoader(fmri, scdir, dist, chunk_length=train_chunk_length)
    loader._split_into_chunks()

    subj = args.subj_id
    ts_emp = loader.all_bold[subj]
    n_roi, n_trs_raw = ts_emp.shape

    # usable length multiple of 100
    num_windows = n_trs_raw // 100
    usable_trs = num_windows * 100
    ts_emp = ts_emp[:, :usable_trs]

    print(f"Norm = {args.norm}")
    sc = loader.get_subject_connectome(subj, norm=args.norm)
    sim = RWWSubjectSimulator(
        sc=sc, node_size=n_roi,
        TP_per_window=train_chunk_length,
        fit_g_EE=True, fit_g_IE=False, fit_g_EI=True,
        use_bifurcation=False, use_fic=False,
        step_size=args.step, g_init=args.g)
    sim.model.to(DEVICE)
    
    # Baseline
    ts_sim, _ = sim.simulate(u=0, num_windows=num_windows, base_window_num=10)
    plot_comparison(ts_sim, ts_emp, full_length=usable_trs, title="base")
    plot_chunkwise_fc(ts_sim, ts_emp, title="base")

    full_emp_FC = torch.tensor(fc(ts_emp), dtype=torch.float32, device=DEVICE)
    cost = CostsRWW(sim.model,
                    use_rate_reg=True, lambda_rate=args.lambda_rate,
                    use_spec_reg=True, lambda_spec=args.lambda_spec,
                    use_disp_reg=True, lambda_disp=args.lambda_disp)
    fitter = Model_fitting(sim.model, cost, device=DEVICE)
    fitter.train(u=0,
                 empFcs=[full_emp_FC],
                 num_epochs=args.epochs,
                 num_windows=num_windows,
                 learningrate=args.lr,
                 early_stopping=True)


    ts_sim, _ = sim.simulate(u=0, num_windows=num_windows, base_window_num=10)
    plot_comparison(ts_sim, ts_emp, full_length=usable_trs, title="final")
    plot_chunkwise_fc(ts_sim, ts_emp, title="final")

if __name__ == "__main__":
    import seaborn as sns
    torch.manual_seed(0); np.random.seed(0)
    main()
