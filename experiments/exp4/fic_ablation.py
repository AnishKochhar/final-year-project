""" Experiment 4 - vector FIC  vs scalar g_IE """

import argparse, json, os, numpy as np, torch, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
import pandas as pd
import re
from collections import defaultdict

from whobpyt.data_loader import BOLDDataLoader, DEVICE
from simulators.rww_simulator import RWWSubjectSimulator
from whobpyt.custom_cost_RWW import CostsRWW
from whobpyt.modelfitting import Model_fitting

def fit_subject(loader, subject_id, use_fic, epochs, lr, g):
    print(f"use_fic = {use_fic}")
    sc = loader.get_subject_connectome(subject_id, norm=True)
    sim = RWWSubjectSimulator(sc, node_size=sc.shape[0], use_fic=use_fic, g_init=g, step_size=0.05)
    sim.model.to(DEVICE)

    emp_FCs = [fc for _, fc in loader.iter_chunks_per_subject(subject_id)]
    cost = CostsRWW(sim.model)
    fit = Model_fitting(sim.model, cost, device=DEVICE)
    fit.train(u=0, empFcs=emp_FCs, num_epochs=epochs, num_windows=1, learningrate=lr, early_stopping=True)

    emp_ts = loader.all_bold[subject_id]
    n_win = emp_ts.shape[1] // sim.model.TRs_per_window
    _, fc_s = sim.simulate(u=0, num_windows=n_win)
    fc_e = np.corrcoef(emp_ts)
    m = np.tril_indices(fc_e.shape[0], -1)
    corr = np.corrcoef(fc_s[m], fc_e[m])[0,1]

    n_epochs = len(fit.trainingStats.loss) # in case of early stopping
    return corr, n_epochs

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="/vol/bitbucket/ank121/fyp/HCP Data")
    p.add_argument("--subjects",  nargs="+", type=int, default=[10, 20, 30, 40, 50])
    p.add_argument("--epochs",    type=int,  default=50)
    p.add_argument("--lr",        type=float, default=0.05)
    p.add_argument("--g",         type=float, default=100)
    args = p.parse_args()

    # distance_matrices_path = os.path.join(args.data_root, "schaefer100_dist.npy")
    # fmri_path = os.path.join(args.data_root, "BOLD Timeseries HCP.mat")
    # sc_dir = os.path.join(args.data_root, "distance_matrices")
    # loader = BOLDDataLoader(fmri_path, sc_dir, distance_matrices_path, chunk_length=50)
    # loader._split_into_chunks()

    stats = {"fic": [(0.313, 29), (0.369, 49), (0.329, 50), (0.382, 50), (0.284, 50)], "scalar": [(0.263, 35), (0.325, 50), (0.244, 50), (0.340, 50), (0.271, 50)]}
    # stats = {"fic": [], "scalar": []}
    # for subject in args.subjects:
    #     print(f"[MAIN] Starting trials for subject = {subject}")
    #     corr_fic, ep = fit_subject(loader, subject, True, args.epochs, args.lr, args.g)
    #     stats["fic"].append((corr_fic, ep))
    #     corr_sca, ep = fit_subject(loader, subject, False, args.epochs, args.lr, args.g)
    #     stats["scalar"].append((corr_sca, ep))

        # print(f"\n[RESULTS] S{subject} FIC = {corr_fic:.3f} | SCALAR = {corr_sca:.3f}\n")

    out = Path("exp4_outputs"); out.mkdir(exist_ok=True)

    # with open(out / "results.json","w") as fp:
    #     json.dump({k: [[float(x) for x in tup] for tup in v] for k, v in stats.items()}, fp, indent=2)


    # def make_plot(metric_idx, ylabel, fname):
    #     vals = {"Vector FIC": [s[metric_idx] for s in stats["fic"]],
    #             "Scalar $g_{IE}$": [s[metric_idx] for s in stats["scalar"]]}
    #     sns.boxplot(data=vals);  sns.swarmplot(data=vals, color=".3", size=6)
    #     plt.ylabel(ylabel); plt.tight_layout()
    #     plt.savefig(out / fname, dpi=300); plt.close()

    # make_plot(0, "FC correlation",          "fc_corr_box.png")
    # make_plot(1, "Epochs to convergence",   "epochs_box.png")

    def km_curve(durations):
        """x=epoch, y=proportion of runs still training."""
        max_e = max(durations)
        surv  = [sum(d > t for d in durations)/len(durations)
                 for t in range(max_e+1)]
        return np.arange(max_e+1), np.asarray(surv)

    epochs_fic    = [e for _, e in stats["fic"]]
    epochs_scalar = [e for _, e in stats["scalar"]]

    x_fic, y_fic       = km_curve(epochs_fic)
    x_sca, y_sca       = km_curve(epochs_scalar)

    plt.figure(figsize=(6, 4))
    plt.step(x_fic, y_fic,  where="post", label="Vector FIC")
    plt.step(x_sca, y_sca,  where="post", label="Scalar $g_{IE}$")
    plt.xlabel("Training epoch");  plt.ylabel("Runs remaining (%)")
    plt.title("Kaplan-Meier convergence curve")
    plt.legend();  plt.tight_layout()
    plt.savefig(out / "km_convergence.png", dpi=300);  plt.close()


    sub_ids   = [10, 20, 30, 40, 50]
    fic_vals  = [fc for fc, _ in stats["fic"]]
    sca_vals  = [fc for fc, _ in stats["scalar"]]

    # df_pair = pd.DataFrame({
    #     "Subject": sub_ids,
    #     "Scalar $g_{IE}$": sca_vals,
    #     "Vector FIC": fic_vals
    # }).set_index("Subject")

    # plt.figure(figsize=(8, 5))
    # for s in df_pair.index:
    #     plt.plot(["Scalar $g_{IE}$", "Vector FIC"],
    #             df_pair.loc[s], marker="o", lw=1.5, alpha=0.8)
    # plt.ylabel("FC correlation (%)")
    # plt.title("Per-subject change\n(Scalar → Vector FIC)")
    # plt.tight_layout()
    # plt.savefig(out / "paired_fc_slope.png", dpi=300)
    # plt.close()

    jit = 0.15  # horizontal jitter
    xf = np.array([e for _, e in stats["fic"]])    + np.random.uniform(-jit, jit, len(sub_ids))
    xs = np.array([e for _, e in stats["scalar"]]) + np.random.uniform(-jit, jit, len(sub_ids))
    yf = np.array(fic_vals)
    ys = np.array(sca_vals)

    plt.figure(figsize=(6, 4))
    plt.scatter(xf, yf, c="#c20044", label="Vector FIC",  marker="o", s=50, edgecolor="k", alpha=0.7)
    plt.scatter(xs, ys, c="#2444ff", label="Scalar $g_{IE}$",  marker="s", s=50, edgecolor="k", alpha=0.7)
    plt.grid(lw=0.3, alpha=0.5)
    plt.xlabel("Epochs to convergence")
    plt.ylabel("FC correlation")
    plt.title("Fit vs. speed")
    plt.legend(framealpha=0.95)
    plt.tight_layout()
    plt.savefig(out / "scatter_fit_vs_speed_pretty.png", dpi=300)
    plt.close()


    # mean_fc_fic  = np.mean([fc for fc,_ in stats["fic"]])
    # mean_fc_sca  = np.mean([fc for fc,_ in stats["scalar"]])
    # with open(out / "table_E4_1.txt", "w") as f:
    #     f.write("Mean FC-corr (Vector FIC)   = {:.4f}\n".format(mean_fc_fic))
    #     f.write("Mean FC-corr (Scalar g_IE) = {:.4f}\n".format(mean_fc_sca))
    #     f.write("n_subjects = {}\n".format(len(stats["fic"])))

    print("Saved to", out)


if __name__ == "__main__":
    main()