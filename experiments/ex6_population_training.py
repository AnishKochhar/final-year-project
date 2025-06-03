""" Experiement 6: POP pre-training -> Subject-specific finetuning """

import argparse, os, time, random, warnings,json, numpy as np, torch
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from whobpyt.data_loader import BOLDDataLoader, DEVICE
from simulators.rww_simulator import RWWSubjectSimulator
from whobpyt.utils.utils_shared import make_shared_pars
from whobpyt.datatypes import par
from whobpyt.hierarchical_population import HierarchicalPopulationFitter
from whobpyt.modelfitting import Model_fitting
from whobpyt.custom_cost_RWW import CostsRWW
from whobpyt.utils.plotter import heatmap_fc

def empirical_windows(loader, sid):
    return [fc for _, fc in loader.iter_chunks_per_subject(sid)]

def full_sim_corr(sim, emp_ts, title="", outdir="exp6_outputs", subdir="plots", subject=None):
    n_win = emp_ts.shape[1] // sim.model.TRs_per_window
    _, fc_sim = sim.simulate(u=0, num_windows=n_win)
    mask = np.tril_indices(fc_sim.shape[0], -1)
    fc_emp = np.corrcoef(emp_ts)
    r = np.corrcoef(fc_sim[mask], fc_emp[mask])[0,1]
    # heatmap_fc(fc_sim, fc_emp, r=r, tag=title, subj=subject, outdir=outdir, subdir=subdir) # Save plot
    return r

def param_vec(sim: RWWSubjectSimulator):
    """ Return  [g , g_EE , g_EI] """
    p = sim.model.params
    return np.array([
        p.g.value().detach().cpu().item(),
        p.g_EE.value().detach().cpu().item(),
        p.g_EI.value().detach().cpu().item()
    ], dtype="float32")



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/vol/bitbucket/ank121/fyp/HCP Data")
    p.add_argument("--subjects", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    p.add_argument("--pop-epochs", type=int, default=20)
    p.add_argument("--ss-epochs",  type=int, default=10)
    p.add_argument("--lr", type=float, default=.05)
    p.add_argument("--lambda-kl", type=float, default=1.0)
    p.add_argument("--windows-per-subj", type=int, default=4,
                help="training minibatch size per subject / epoch")
    p.add_argument("--seed", type=int, default=0)

    args = p.parse_args()
    
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out_dir = Path("exp6_outputs"); out_dir.mkdir(exist_ok=True)

    fmri = os.path.join(args.data_root, "BOLD Timeseries HCP.mat")
    scdir = os.path.join(args.data_root, "distance_matrices")
    dist  = os.path.join(args.data_root, "schaefer100_dist.npy")
    loader = BOLDDataLoader(fmri, scdir, dist, chunk_length=30)
    loader._split_into_chunks()


    ## Population Fitting
    shared = make_shared_pars()
    sims, emp_banks = [], []
    for sid in args.subjects:
        sc = loader.get_subject_connectome(sid, norm=True)
        sims.append(RWWSubjectSimulator(sc, node_size=sc.shape[0], TP_per_window=loader.chunk_length, use_fic=True, g_init=50, \
                                        step_size=0.05, shared_params=shared))
        emp_banks.append(empirical_windows(loader, sid))
    
    popfit = HierarchicalPopulationFitter(sims, emp_banks, shared,
                                          lr_subject=args.lr, lr_mean=args.lr, lambda_kl=args.lambda_kl)
    t0 = time.time()
    for ep in range(args.pop_epochs):
        _ = popfit.train_epoch(windows_per_subj=args.windows_per_subj, clip_grad=1.0, log_every=True)


    print(f"POP fitting finished in {(time.time()-t0)/60:.1f} min\n")
    r_pop, deltas = {}, {}
    for sid, sim in zip(args.subjects, sims):
        r_pop[sid] = full_sim_corr(sim, loader.all_bold[sid], subject=sid, outdir="exp6_outputs", subdir="plots", title=f"S{sid} POP fit")
        deltas[sid] = param_vec(sim) - np.array([shared[k].value().item() for k in ("g_mean", "g_EE_mean", "g_EI_mean")])
        print(f"S{sid} POP fit deltas: {deltas[sid]}")

    ## Subject-specific finetune
    r_ss = {}
    for sid in args.subjects:
        sc = loader.get_subject_connectome(sid, norm=True)
        # Start from shared learned above
        init_pars = {
            k: par(v.value().detach().clone(), fit_par=True, device=DEVICE)
            for k, v in shared.items()
        }
        sim = RWWSubjectSimulator(sc, node_size=sc.shape[0], TP_per_window=loader.chunk_length, use_fic=True, g_init=50, step_size=0.05, shared_params=init_pars) 
        cost = CostsRWW(sim.model)
        fit  = Model_fitting(sim.model, cost, device=DEVICE)

        fit.train(u=0, empFcs=empirical_windows(loader, sid), num_epochs=args.ss_epochs, num_windows=1,
                  learningrate=args.lr, early_stopping=True)

        r_ss[sid] = full_sim_corr(sim, loader.all_bold[sid], subject=sid, outdir="exp6_outputs", subdir="plots", title=f"S{sid} SS fit")
        deltas[sid] = param_vec(sim) - \
                np.array([shared[k].value().item()
                        for k in ("g_mean","g_EE_mean","g_EI_mean")])

        print(f"[Subject {sid}] Δr = {r_ss[sid]-r_pop[sid]:+.4f}")

    results = dict(
        meta = vars(args),
        r_pop = r_pop,
        r_ss  = r_ss,
        delta_params = {sid: d.tolist() for sid, d in deltas.items()}
    )
    json.dump(results, open(out_dir/"metrics.json", "w"), indent=2)


    # Plot results
    ids = list(args.subjects)
    pop_vals = np.array([r_pop[i] for i in ids])
    ss_vals  = np.array([r_ss[i]  for i in ids])
    delta_r  = ss_vals - pop_vals
    delta_norm = np.array([np.linalg.norm(deltas[i]) for i in ids])
    mean_vals = 0.5 * (pop_vals + ss_vals)

    # Scatter POP vs SS
    plt.figure(figsize=(4,4))
    plt.scatter(pop_vals, ss_vals, c="tab:blue")
    lims = [min(pop_vals.min(), ss_vals.min())-0.02,
            max(pop_vals.max(), ss_vals.max())+0.02]
    plt.plot(lims, lims, "--", color="gray")
    plt.xlabel(r"$r_{\mathrm{POP}}$")
    plt.ylabel(r"$r_{\mathrm{SS}}$")
    plt.title("FC correlation: POP vs SS")
    plt.tight_layout()
    plt.savefig(out_dir/"scatter_pop_vs_ss.png", dpi=300)
    plt.close()

    # Histogram of change in corrs
    plt.figure(figsize=(4,3))
    plt.hist(delta_r, bins="auto", color="tab:orange", alpha=.8, rwidth=.9)
    plt.axvline(0, color="gray", linestyle="--")
    plt.xlabel(r"$\Delta r = r_{\mathrm{SS}}-r_{\mathrm{POP}}$")
    plt.title("Improvement after fine-tune")
    plt.tight_layout()
    plt.savefig(out_dir/"hist_delta_r.png", dpi=300)
    plt.close()

    # Bland-Altman
    plt.figure(figsize=(4,3))
    plt.scatter(mean_vals, delta_r, color="tab:green")
    plt.axhline(delta_r.mean(), color="k", linestyle="--")
    plt.xlabel("mean( POP , SS )")
    plt.ylabel(r"$\Delta r$")
    plt.title("Bland-Altman plot")
    plt.tight_layout()
    plt.savefig(out_dir/"bland_altman.png", dpi=300)
    plt.close()

    # Personalization gain vs parameter distance
    plt.figure(figsize=(4,3))
    plt.scatter(delta_norm, delta_r, color="tab:red")
    m, b = np.polyfit(delta_norm, delta_r, 1)
    xfit = np.linspace(delta_norm.min(), delta_norm.max(), 100)
    plt.plot(xfit, m*xfit + b, "--", color="black")
    plt.xlabel(r"$\|\theta^{(s)}-\bar\theta\|_2$ before fine-tune")
    plt.ylabel(r"$\Delta r$")
    plt.title("Personalization gain vs parameter distance")
    plt.tight_layout()
    plt.savefig(out_dir/"dr_vs_dtheta.png", dpi=300)
    plt.close()

    print("\nFigures saved to", out_dir.resolve())


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
