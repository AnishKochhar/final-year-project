""" Grid search (G, G_EE) against 5-node ground truth """

import argparse, json, itertools, time
from pathlib import Path
import numpy as np, torch, matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from whobpyt.data_loader import DEVICE
from simulators.rww_simulator import RWWSubjectSimulator
from whobpyt.custom_cost_RWW import CostsRWW
from whobpyt.modelfitting import Model_fitting
from whobpyt.utils.plotter import heatmap_fc

SAVE_DIR = Path("exp1_outputs"); SAVE_DIR.mkdir(exist_ok=True, parents=True)

def static_fc(ts, drop=10):
    """ Pearson FC of BOLD (N,T) after removing `drop` leading TRs """
    fc = np.corrcoef(ts[:, drop:])
    m  = np.tril_indices(fc.shape[0], -1)
    return fc, fc[m]

def dynamic_fc(ts, win=50, stride=25):
    """ Sliding-window FC. Return matrix (W, E) where E=#edges """
    N, T = ts.shape
    idx  = np.tril_indices(N, -1)
    slices = range(0, T - win + 1, stride)
    out = []
    for s in slices:
        w_fc = np.corrcoef(ts[:, s:s+win])
        out.append(w_fc[idx])
    return np.vstack(out)          # shape (n_windows, n_edges)

def dynfc_similarity(ts_cand, ts_gt, win=50, stride=25):
    """ Return per-window FC correlation and its mean """
    dFC_c = dynamic_fc(ts_cand, win, stride)   # (W, E)
    dFC_g = dynamic_fc(ts_gt,   win, stride)
    corrs = [ np.corrcoef(vc, vg)[0,1] for vc, vg in zip(dFC_c, dFC_g) ]
    return np.asarray(corrs), np.mean(corrs)


def fc_corr_static(ts_candidate, vec_gt):
    _, vec = static_fc(ts_candidate)
    return np.corrcoef(vec, vec_gt)[0,1]

def run_once(sc, g, g_ee, g_ei, step, tp, n_win, base, fic, fc_emp):
    sim = RWWSubjectSimulator(sc, node_size=5, g_init=g,
                              step_size=step, TP_per_window=tp,
                              use_fic=fic)
    sim.model.to(DEVICE)
    sim.model.params.g_EE.val.data  = torch.tensor(g_ee, device=DEVICE)
    sim.model.params.g_EI.val.data  = torch.tensor(g_ei, device=DEVICE)

    fitter = Model_fitting(sim.model, CostsRWW(sim.model), device=DEVICE)
    _, fc_sim = fitter.simulate(u=0, num_windows=n_win, base_window_num=base, transient_num=10)
    
    corr = fitter.evaluate([fc_emp], [fc_sim])
    return corr
    


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--step",    type=float, default=0.05)
    p.add_argument("--tp",      type=int,   default=50)
    p.add_argument("--windows", type=int,   default=80)
    p.add_argument("--base",    type=int,   default=10)
    p.add_argument("--rep",     type=int,   default=3, help="replicates/point")
    args = p.parse_args()

    gt_fc = np.load(SAVE_DIR/"gt_fc.npy")
    sc_np = np.load(SAVE_DIR/"ring_sc.npy")
    sc    = torch.tensor(sc_np, dtype=torch.float32, device=DEVICE)

    # g_grid   = np.arange(10, 151, 70)
    # gee_grid = np.arange(1.5, 5.6, 2.0)
    g_grid   = np.arange(10, 151, 10)
    gee_grid = np.arange(1, 6.1, 0.5)
    res_mean = np.zeros((len(gee_grid), len(g_grid)))

    G_EI_STAR = 0.42

    for i, gee in enumerate(gee_grid):
        for j, g in enumerate(g_grid):
            print(f"Testing g={g} g_EE={gee}")
            corrs = []
            for seed in range(args.rep):
                np.random.seed(seed); torch.manual_seed(seed)
                corr = run_once(sc, g, gee, g_ei=G_EI_STAR, step=args.step, tp=args.tp, n_win=args.windows, base=args.base, fic=False, fc_emp=gt_fc)
                corrs.append(corr)
            res_mean[i,j] = np.mean(corrs)
            print(f"gee={gee:4.2f}  g={g:6.1f}  corr={res_mean[i,j]:.4f}")

    np.save(SAVE_DIR/"grid_fc_corr.npy", res_mean)
    json.dump({"g": g_grid.tolist(), "g_EE": gee_grid.tolist()}, open(SAVE_DIR/"grid_axes.json","w"), indent=2)
    
    # FC correlation heatmap (2D + colour)
    plt.figure(figsize=(6,4))
    sns.heatmap(res_mean, cmap="viridis", xticklabels=g_grid, yticklabels=gee_grid, cbar_kws={"label":"mean FC-corr"})
    plt.scatter(list(g_grid).index(80)+0.5,
                list(gee_grid).index(3.5)+0.5,
                c="red", marker="x", s=60)
    plt.xlabel("g"); plt.ylabel("g_EE")
    plt.tight_layout(); plt.savefig(SAVE_DIR/"heatmap_grid.png", dpi=300)
    plt.close()

    # slice through true g_EE
    idx = list(gee_grid).index(3.5)
    plt.figure(); plt.plot(g_grid, res_mean[idx], "-o")
    plt.axvline(80, ls="--", c="red"); plt.ylabel("FC-corr"); plt.xlabel("g")
    plt.title("g_EE = 3.5")
    plt.tight_layout(); plt.savefig(SAVE_DIR/"slice_gEE_star.png", dpi=300)
    plt.close()

    # demo_g = [60, 80, 100]
    # win, stride = 50, 25
    # for g in demo_g:
    #     ts = run_once(sc, g, 3.5, g_ei=G_EI_STAR, step=args.step, tp=args.tp, n_win=20, base=args.base)
    #     corrs, avg = dynfc_similarity(ts, gt_ts, win=50, stride=25)

    #     plt.figure(figsize=(4,2.5))
    #     plt.plot(corrs, marker='o'); plt.ylim(-1,1)
    #     plt.axhline(avg, ls='--', c='red', label=f"mean={avg:.2f}")
    #     plt.title(f"dyn-FC corr vs GT | g={g}")
    #     plt.xlabel("window"); plt.ylabel("corr")
    #     plt.legend(loc="lower right", fontsize=7)
    #     plt.tight_layout()
    #     plt.savefig(SAVE_DIR/f"dynfc_corr_g{g}.png", dpi=300); plt.close()


if __name__ == "__main__":
    main()
