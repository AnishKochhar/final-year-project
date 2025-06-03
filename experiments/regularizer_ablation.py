""" Hyperparameter search over rate / spectral-slope / dispersion regularizers """

import argparse, itertools, json, os, random, warnings
from typing import Sequence
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="whitegrid")

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from whobpyt.data_loader import BOLDDataLoader, DEVICE
from simulators.rww_simulator import RWWSubjectSimulator
from whobpyt.custom_cost_RWW   import CostsRWW
from whobpyt.modelfitting      import Model_fitting


def run_single_seed(loader, subject: int, lambdas: Sequence[float], \
                    *, epochs: int, lr: float, g: float, step_size: float, \
                        train_windows: int, eval_windows: int, rng_seed: int = 0) -> float:
    """ Train once and return FC correlation """
    lam_rate, lam_spec, lam_disp = lambdas
    np.random.seed(rng_seed); torch.manual_seed(rng_seed)

    sc = loader.get_subject_connectome(subject, norm=True)
    sim = RWWSubjectSimulator(sc, TP_per_window=loader.chunk_length, node_size=sc.shape[0], use_fic=True, g_init=g, step_size=step_size)
    sim.model.to(DEVICE)

    emp_FCs = [fc for _, fc in loader.iter_chunks_per_subject(subject)]
    if train_windows and len(emp_FCs) > train_windows:
        emp_FCs = random.sample(emp_FCs, k=train_windows)

    cost = CostsRWW(sim.model,
                    use_rate_reg = lam_rate  > 0, lambda_rate = lam_rate,
                    use_spec_reg = lam_spec  > 0, lambda_spec = lam_spec,
                    use_disp_reg = lam_disp  > 0, lambda_disp = lam_disp)

    fit = Model_fitting(sim.model, cost, device=DEVICE)
    fit.train(u=0, empFcs=emp_FCs, num_epochs=epochs, num_windows=1,
              learningrate=lr, max_chunks=20, early_stopping=True)

    emp_ts = loader.all_bold[subject]
    # n_win  = emp_ts.shape[1] // sim.model.TRs_per_window
    full_windows = emp_ts.shape[1] // sim.model.TRs_per_window
    if eval_windows == 0 or eval_windows >= full_windows:
        start = 0; n_win = full_windows
    else:
        start = full_windows - eval_windows
        n_win = eval_windows
    _, fc_sim = sim.simulate(u=0, num_windows=n_win, base_window_num=start)
    ts_seg   = emp_ts[:, start * sim.model.TRs_per_window:]
    fc_emp    = np.corrcoef(ts_seg)

    mask = np.tril_indices(fc_emp.shape[0], -1)
    corr = float(np.corrcoef(fc_sim[mask], fc_emp[mask])[0, 1])
    print(f"CORR = {corr:.4f} [{lam_rate:.3f}, {lam_spec:.3f}, {lam_disp:.3f}]")
    return corr

def evaluate(loader, subject, lambdas, *,
             reps=3, epochs=25, lr=0.05, g=85, step_size=0.01,
             train_windows, eval_windows, parallel_seeds):
    """ Average correlation across `reps` seeds for robustness """
    def _call(seed):
         return run_single_seed(loader, subject, lambdas,
                                epochs=epochs, lr=lr, g=g, step_size=step_size,
                                rng_seed=seed,
                                train_windows=train_windows,
                                eval_windows=eval_windows)

    if parallel_seeds > 1:
        with ProcessPoolExecutor(max_workers=parallel_seeds) as ex:
            corrs = list(ex.map(_call, range(reps)))
    else:
        corrs = [_call(seed) for seed in range(reps)]
    return np.mean(corrs), corrs


def build_objective(loader, subject, args):
    def objective(trial: optuna.trial.Trial) -> float:
        lam_rate = trial.suggest_float("lambda_rate", 0.0, 0.2)
        lam_spec = trial.suggest_float("lambda_spec", 0.0, 0.2)
        lam_disp = trial.suggest_float("lambda_disp", 0.0, 0.2)

        mean_corr, corr_list = evaluate(
            loader, subject, (lam_rate, lam_spec, lam_disp),
            reps=args.reps, epochs=args.epochs, lr=args.lr,
            g=args.g, step_size=args.step, train_windows=args.train_windows,
            eval_windows=args.eval_windows, parallel_seeds=args.parallel_seeds)

        # keep raw list for later diagnostics
        trial.set_user_attr("corr_list", corr_list)
        return mean_corr
    return objective

def binary_ablation(loader, subject, args, λ_star):
    """ Eight on/off combinations at fixed optimal lambda values """
    combos = list(itertools.product([0.0, λ_star[0]],
                                    [0.0, λ_star[1]],
                                    [0.0, λ_star[2]]))
    results, raw_corrs = {}, {}
    for lam in combos:
        mean_corr, corr_list = evaluate(loader, subject, lam,
                                        reps=args.reps, epochs=args.epochs,
                                        lr=args.lr, g=args.g,
                                        step_size=args.step)
        results[lam]     = mean_corr
        raw_corrs[lam]   = corr_list
        print(f"[GRID] {lam}: mean FC-corr = {mean_corr:.3f}")
    return results, raw_corrs

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-root", type=str, default="/vol/bitbucket/ank121/fyp/HCP Data")
    p.add_argument("--subject",   type=int, default=0, help="HCP subject index")
    p.add_argument("--epochs",    type=int, default=25)
    p.add_argument("--lr",        type=float, default=0.01)
    p.add_argument("--g",         type=float, default=85)
    p.add_argument("--step",      type=float, default=0.01)
    p.add_argument("--chunk",     type=int,   default=30)
    p.add_argument("--reps",      type=int,   default=3, help="random seeds per eval")
    p.add_argument("--train-windows",  type=int, default=20, 
                   help="max empirical FC chunks used per epoch (0 = all)")
    p.add_argument("--eval-windows",   type=int, default=20, 
                   help="number of unseen windows for test FC (0 = full TS)")
    p.add_argument("--parallel-seeds", type=int, default=1,
                   help="how many seeds to run concurrently")

    # Optuna
    p.add_argument("--trials",    type=int, default=200)
    p.add_argument("--study-name", type=str, default="e3_optuna")
    p.add_argument("--seed",      type=int, default=1234)

    args = p.parse_args()

    fmri_path      = os.path.join(args.data_root, "BOLD Timeseries HCP.mat")
    sc_dir         = os.path.join(args.data_root, "distance_matrices")
    dist_mat       = os.path.join(args.data_root, "schaefer100_dist.npy")
    loader         = BOLDDataLoader(fmri_path, sc_dir, dist_mat,
                                    chunk_length=args.chunk)
    loader._split_into_chunks()


    ## 1. Optuna Search
    optuna.seed = args.seed
    sampler = TPESampler(seed=args.seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    storage = f"sqlite:///{args.study_name}.db"
    study = optuna.create_study(direction="maximize",
                                sampler=sampler, pruner=pruner,
                                study_name=args.study_name,
                                storage=storage,
                                load_if_exists=True)

    objective = build_objective(loader, args.subject, args)
    print(f"[INFO] Starting Optuna search ({args.trials} trials)")
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    best = study.best_params
    best_val = study.best_value
    with open(f"optuna_{args.study_name}.json", "w") as fp:
        json.dump({"best_params": best, "best_value": best_val}, fp, indent=2)
    print("\n[OPTUNA] Best λ's:", best, " -> ", f"{best_val:.3f}")


    ## 2. Binary ablation grid
    λ_star = (best["lambda_rate"], best["lambda_spec"], best["lambda_disp"])
    grid_res, grid_raw = binary_ablation(loader, args.subject, args, λ_star)

    Path("plots").mkdir(exist_ok=True)
    Path("exp3_outputs").mkdir(exist_ok=True)

    # Save grid results
    json.dump({str(k): float(v) for k, v in grid_res.items()},
              open("exp3_outputs/results.json", "w"), indent=2)
    


    ## 3. Plots
    # 3-D scatter of Optuna trials
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(6, 4))
    ax  = fig.add_subplot(111, projection='3d')
    xs, ys, zs, cs = [], [], [], []
    for t in study.trials:
        xs.append(t.params["lambda_rate"])
        ys.append(t.params["lambda_spec"])
        zs.append(t.params["lambda_disp"])
        cs.append(t.value)
    sc = ax.scatter(xs, ys, zs, c=cs, cmap="viridis", s=120, alpha=0.85)
    cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.1, label="FC correlation")
    ax.set_xlabel(r"$\lambda_\mathrm{rate}$")
    ax.set_ylabel(r"$\lambda_\mathrm{spec}$")
    ax.set_zlabel(r"$\lambda_\mathrm{disp}$")
    ax.set_title("Fig E3-1. Optuna search – FC correlation")
    ax.view_init(elev=20, azim=210)
    # highlight best
    ax.scatter(best["lambda_rate"], best["lambda_spec"], best["lambda_disp"],
               s=250, edgecolor='k', facecolor='none', linewidth=2)
    plt.tight_layout()
    fig.savefig("exp3_ouputs/fig_e3_scatter3d.png", dpi=300)
    plt.close(fig)

    # Strip plot (binary)
    x_labels, y = [], []
    for lam, mean_corr in grid_res.items():
        x_labels.append(f"{int(lam[0]>0)}{int(lam[1]>0)}{int(lam[2]>0)}")
        y.append(mean_corr)

    fig, ax = plt.subplots(figsize=(6, 3))
    sns.stripplot(x=x_labels, y=y, size=8, ax=ax)
    ax.set_ylabel("FC correlation")
    ax.set_xlabel("Regulariser on/off pattern\n( rate  spec  disp )")
    plt.subplots_adjust(bottom=0.18)   # prevent label cut-off
    ax.set_title("Fig E3-2. Main effects (binary toggle)")
    fig.savefig("exp3_outputs/fig_e3_strip.png", dpi=300)
    plt.close(fig)

    # simple 2-level ANOVA with statsmodels
    try:
        import pandas as pd, statsmodels.formula.api as smf
        df = pd.DataFrame({
            "corr":  y,
            "rate":  [lbl[0] for lbl in x_labels],
            "spec":  [lbl[1] for lbl in x_labels],
            "disp":  [lbl[2] for lbl in x_labels],
        })
        model = smf.ols("corr ~ rate + spec + disp + rate:spec + rate:disp + spec:disp", data=df).fit()
        anova = smf.stats.anova_lm(model, typ=2)
        anova.to_csv("json/anova.csv")
        print("\n[ANOVA]\n", anova)
    except ModuleNotFoundError:
        print("[WARN] statsmodels not installed - skipping ANOVA")

    print("[INFO] Finished!")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        main()



# def run_fit(loader, subject, lambdas, epochs, lr, g = 50, step = 0.05):
#     print(f'Testing: {lambdas}')
#     lam_rate, lam_spec, lam_disp = lambdas
#     sc = loader.get_subject_connectome(subject, norm=True)
#     sim = RWWSubjectSimulator(sc, node_size=sc.shape[0], use_fic=True, g_init=g, step_size=step)
#     sim.model.to(DEVICE)

#     emp_FCs = [fc for _, fc in loader.iter_chunks_per_subject(subject)]
    
#     cost = CostsRWW(sim.model,
#                 use_rate_reg = lam_rate>0, lambda_rate = lam_rate,
#                 use_spec_reg = lam_spec>0, lambda_spec = lam_spec,
#                 use_disp_reg = lam_disp>0, lambda_disp = lam_disp)

#     fit = Model_fitting(sim.model, cost, device=DEVICE)
#     fit.train(u=0, empFcs=emp_FCs, num_epochs=epochs, num_windows=1, learningrate=lr, max_chunks=20, early_stopping=True)

#     emp_ts = loader.all_bold[subject]
#     n_win = emp_ts.shape[1] // sim.model.TRs_per_window
#     _, fc_s = sim.simulate(u=0, num_windows=n_win)
#     fc_e = np.corrcoef(emp_ts)
#     m = np.tril_indices(fc_e.shape[0], -1)
#     return np.corrcoef(fc_s[m], fc_e[m])[0,1], fit.trainingStats.loss

# def run_lambda(loader, subject, lambdas, epochs, lr, g = 50, step = 0.05, its=3):
#     corrs, loss_curves = [], []
#     for rep in range(its):
#         np.random.seed(rep); torch.manual_seed(rep)
#         corr, loss_curve = run_fit(loader, subject, lambdas, epochs=epochs, lr=lr, g=g, step=step)
#         corrs.append(corr)
#         if len(loss_curve) < epochs:
#             final_val = loss_curve[-1]; padded_curve = loss_curve + [final_val] * (epochs - len(loss_curve))
#         else: 
#             padded_curve = loss_curve
#         loss_curves.append(padded_curve)
    
#     return corrs, np.mean(loss_curves, axis=0)

# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument("--data-root", type=str, default="/vol/bitbucket/ank121/fyp/HCP Data")
#     p.add_argument("--g",         type=float, default=85)
#     p.add_argument("--step",      type=float, default=0.01)
#     p.add_argument("--epochs",    type=int, default=25)
#     p.add_argument("--lr",        type=float, default=0.05)
#     p.add_argument("--chunk",     type=int,   default=30)
#     p.add_argument("--subject",   type=int,  default=0)
#     args = p.parse_args()


#     distance_matrices_path = os.path.join(args.data_root, "schaefer100_dist.npy")
#     fmri_path = os.path.join(args.data_root, "BOLD Timeseries HCP.mat")
#     sc_dir = os.path.join(args.data_root, "distance_matrices")
#     loader = BOLDDataLoader(fmri_path, sc_dir, distance_matrices_path, chunk_length=args.chunk)
#     loader._split_into_chunks()

#     combos = list(itertools.product([0.0, .1], [0.0, .05], [0.0, .05]))  # (λ_rate, λ_spec, λ_disp)
#     # combos = list(itertools.product([.1], [.05], [0.0, .05]))  # (λ_rate, λ_spec, λ_disp)
#     results, losses, raw_corrs = {}, {}, {}  
#     for lam in combos:
#         corr_list, loss_curve = run_lambda(loader, args.subject, lam, epochs=args.epochs, lr=args.lr, g=args.g, step=args.step)
#         results[lam] = np.mean(corr_list)
#         raw_corrs[lam] = corr_list
#         losses[lam] = loss_curve

#         print(f"{lam}: FC-corr = {np.mean(corr_list):.3f}")

#     save_dir = "exp3_outputs_3trials"; Path(save_dir).mkdir(exist_ok=True)
#     json.dump({str(k):float(v) for k,v in results.items()}, open(f"{save_dir}/results.json","w"), indent=2)

#     # 1. Box / Strip
#     plt.figure(figsize=(6,3))
#     x_labels, y = [], []
#     for lam,corr in results.items():
#         x_labels.append(f"{int(lam[0]>0)}{int(lam[1]>0)}{int(lam[2]>0)}")
#         y.append(corr)
#     sns.stripplot(x=x_labels, y=y, size=8)
#     plt.ylabel("FC correlation"); plt.xlabel("Regulariser combination (firing rate, spectral slope, dispersion)")
#     plt.savefig(f"{save_dir}/fc_corr_strip.png", dpi=300); plt.close()

#     means, stds, labels = [], [], []
#     for lam in combos:
#         c = raw_corrs[lam]
#         means.append(np.mean(c))
#         stds.append(np.std(c))
#         labels.append(f"({lam[0]}, {lam[1]}, {lam[2]})")

#     plt.figure(figsize=(6,4))
#     plt.errorbar(labels, means, yerr=stds, fmt='o', capsize=4)
#     plt.ylabel("FC correlation"); plt.xlabel("Regulariser combination (rate spec disp)")
#     plt.savefig(f"{save_dir}/fc_corr_errorbars.png", dpi=300); plt.close()


#     # 2. 3D Bar
#     from matplotlib import cm
#     from matplotlib.ticker import MaxNLocator, FormatStrFormatter

#     fig = plt.figure(figsize=(6,4))
#     ax = fig.add_subplot(111, projection='3d')
#     dx, dy = 0.08, 0.04

#     # Normalize FC correlation for color mapping
#     corr_vals = list(results.values())
#     norm = plt.Normalize(min(corr_vals), max(corr_vals))
#     cmap = cm.get_cmap("viridis")

#     for (lr, ls, ld), corr in results.items():
#         color = cmap(norm(corr))
#         ax.bar3d(lr, ls, ld, dx, dy, corr, color=color, shade=True)

#     # Axes and formatting
#     ax.set_xlabel("λ_rate")
#     ax.set_ylabel("λ_spec")
#     ax.set_zlabel("λ_disp")
#     ax.set_title("FC-corr height; bar depth = λ_disp")

#     # Simplify ticks
#     ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
#     ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#     ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
#     ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#     ax.zaxis.set_major_locator(MaxNLocator(nbins=4))
#     ax.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))

#     # Colorbar
#     sm = cm.ScalarMappable(cmap=cmap, norm=norm)
#     sm.set_array([])
#     fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.1, label="FC correlation")

#     plt.savefig(f"{save_dir}/fc_corr_3d.png", dpi=300)
#     plt.close()


#     plt.figure()
#     for lam, lcurve in losses.items():
#         plt.plot(lcurve, label=f"{lam}")
#     plt.legend(fontsize=7); plt.xlabel("Window"); plt.ylabel("Loss")
#     plt.tight_layout(); plt.savefig(f"{save_dir}/loss_curves.png", dpi=300); plt.close()



#     fig = plt.figure(figsize=(6,4))
#     ax = fig.add_subplot(111, projection='3d')
#     # Gather all points and FC correlation values
#     xs, ys, zs, cs = zip(*[(lr, ls, ld, corr) for (lr, ls, ld), corr in results.items()])
#     sc = ax.scatter(xs, ys, zs, c=cs, cmap="viridis", s=80)
#     fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.1, label="FC correlation")
#     ax.set_xlabel("λ_rate"); ax.set_ylabel("λ_spec"); ax.set_zlabel("λ_disp")
#     plt.savefig(f"{save_dir}/fc_corr_3dscatter.png", dpi=300)
#     plt.close()



# if __name__ == '__main__':
#     main()
