""" Ablation of rate / spectral-slope / dispersion regularizers """

import argparse, itertools, json, torch, os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="whitegrid")

from whobpyt.data_loader import BOLDDataLoader, DEVICE
from simulators.rww_simulator import RWWSubjectSimulator
from whobpyt.custom_cost_RWW   import CostsRWW
from whobpyt.modelfitting      import Model_fitting
from whobpyt.utils.plotter     import heatmap_fc


def run_fit(loader, subject, lambdas, epochs, lr, g = 50, step = 0.05):
    print(f'Testing: {lambdas}')
    lam_rate, lam_spec, lam_disp = lambdas
    sc = loader.get_subject_connectome(subject, norm=True)
    sim = RWWSubjectSimulator(sc, node_size=sc.shape[0], use_fic=True, g_init=g, step_size=step)
    sim.model.to(DEVICE)

    emp_FCs = [fc for _, fc in loader.iter_chunks_per_subject(subject)]
    
    cost = CostsRWW(sim.model,
                use_rate_reg = lam_rate>0, lambda_rate = lam_rate,
                use_spec_reg = lam_spec>0, lambda_spec = lam_spec,
                use_disp_reg = lam_disp>0, lambda_disp = lam_disp)

    fit = Model_fitting(sim.model, cost, device=DEVICE)
    fit.train(u=0, empFcs=emp_FCs, num_epochs=epochs, num_windows=1, learningrate=lr, max_chunks=20, early_stopping=True)

    emp_ts = loader.all_bold[subject]
    n_win = emp_ts.shape[1] // sim.model.TRs_per_window
    _, fc_s = fit.simulate(u=0, num_windows=n_win)
    fc_e = np.corrcoef(emp_ts)
    m = np.tril_indices(fc_e.shape[0], -1)
    return np.corrcoef(fc_s[m], fc_e[m])[0,1], fit.trainingStats.loss

def run_lambda(loader, subject, lambdas, epochs, lr, g = 50, step = 0.05, its=3):
    corrs, loss_curves = [], []
    for rep in range(its):
        np.random.seed(rep); torch.manual_seed(rep)
        corr, loss_curve = run_fit(loader, subject, lambdas, epochs=epochs, lr=lr, g=g, step=step)
        corrs.append(corr); loss_curves.append(loss_curve)
    
    return corrs, np.mean(loss_curves, axis=0)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="/vol/bitbucket/ank121/fyp/HCP Data")
    p.add_argument("--g",         type=float, default=85)
    p.add_argument("--step",      type=float, default=0.01)
    p.add_argument("--epochs",    type=int, default=25)
    p.add_argument("--lr",        type=float, default=0.05)
    p.add_argument("--subject",   type=int,  default=0)
    args = p.parse_args()


    distance_matrices_path = os.path.join(args.data_root, "schaefer100_dist.npy")
    fmri_path = os.path.join(args.data_root, "BOLD Timeseries HCP.mat")
    sc_dir = os.path.join(args.data_root, "distance_matrices")
    loader = BOLDDataLoader(fmri_path, sc_dir, distance_matrices_path, chunk_length=50)
    loader._split_into_chunks()

    combos = list(itertools.product([0.0, .1], [0.0, .05], [0.0, .05]))  # (λ_rate, λ_spec, λ_disp)
    results, losses, raw_corrs = {}, {}, {}  
    for lam in combos:
        corr_list, loss_curve = run_lambda(loader, args.subject, lam, epochs=args.epochs, lr=args.lr, g=args.g, step=args.step)
        results[lam] = np.mean(corr_list)
        raw_corrs[lam] = corr_list
        losses[lam] = loss_curve

        print(f"{lam}  →  FC-corr = {np.mean(corr_list):.3f}")

    save_dir = "exp3_outputs_3trials"; Path(save_dir).mkdir(exist_ok=True)
    json.dump({str(k):float(v) for k,v in results.items()}, open(f"{save_dir}/results.json","w"), indent=2)

    # 1. Box / Strip
    plt.figure(figsize=(6,3))
    x_labels, y = [], []
    for lam,corr in results.items():
        x_labels.append(f"{int(lam[0]>0)}{int(lam[1]>0)}{int(lam[2]>0)}")
        y.append(corr)
    sns.stripplot(x=x_labels, y=y, size=8)
    plt.ylabel("FC correlation"); plt.xlabel("Regulariser combination (firing rate, spectral slope, dispersion)")
    plt.savefig(f"{save_dir}/fc_corr_strip.png", dpi=300); plt.close()

    means, stds, labels = [], [], []
    for lam in combos:
        c = raw_corrs[lam]
        means.append(np.mean(c))
        stds.append(np.std(c))
        labels.append(f"{int(lam[0]>0)}{int(lam[1]>0)}{int(lam[2]>0)}")

    plt.figure(figsize=(6,3))
    plt.errorbar(labels, means, yerr=stds, fmt='o', capsize=4)
    plt.ylabel("FC correlation"); plt.xlabel("Regulariser combination (rate spec disp)")
    plt.savefig(f"{save_dir}/fc_corr_errorbars.png", dpi=300); plt.close()


    # 2. 3D Bar
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111, projection='3d')
    dx, dy = 0.08, 0.04
    for (lr, ls, ld), corr in results.items():
        ax.bar3d(lr, ls, ld, dx, dy, corr, shade=True)
    ax.set_xlabel("λ_rate"); ax.set_ylabel("λ_spec"); ax.set_zlabel("λ_disp")
    ax.set_title("FC-corr height; bar depth = λ_disp")
    plt.savefig(f"{save_dir}/fc_corr_3d.png", dpi=300); plt.close()

    # from mpl_toolkits.mplot3d import Axes3D       # noqa: F401
    # fig = plt.figure(figsize=(5,4)); ax = fig.add_subplot(111, projection='3d')
    # for (lr,ls,ld), corr in results.items():
    #     ax.bar3d(lr, ls, 0, 0.08,0.04, corr, shade=True)
    # ax.set_xlabel("λ_rate"); ax.set_ylabel("λ_spec"); ax.set_zlabel("FC-corr")
    # plt.savefig(f"{save_dir}/fc_corr_3d.png", dpi=300); plt.close()

    plt.figure()
    for lam, lcurve in losses.items():
        plt.plot(lcurve, label=f"{lam}")
    plt.legend(fontsize=7); plt.xlabel("Window"); plt.ylabel("Loss")
    plt.tight_layout(); plt.savefig(f"{save_dir}/loss_curves.png", dpi=300); plt.close()

    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111, projection='3d')
    for (lr, ls, ld), corr in results.items():
        ax.scatter(lr, ls, ld, c=corr, cmap="viridis", s=80)
    ax.set_xlabel("λ_rate"); ax.set_ylabel("λ_spec"); ax.set_zlabel("λ_disp")
    plt.savefig(f"{save_dir}/fc_corr_3dscatter.png", dpi=300); plt.close()


if __name__ == '__main__':
    main()
