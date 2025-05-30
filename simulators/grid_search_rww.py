# Anish Kochhar, May 2025

import itertools, json, argparse, time
import numpy as np, torch, matplotlib.pyplot as plt

from whobpyt.data_loader import BOLDDataLoader, DEVICE
from whobpyt.utils.plotter import heatmap_fc, scatter_fc, timeseries, laplacian, heatmap_sc
from simulators.rww_simulator import RWWSubjectSimulator
from whobpyt.custom_cost_RWW import CostsRWW
from whobpyt.modelfitting import Model_fitting

def run_single(loader, subject, g, step, chunk, epochs=25, lr=0.05, args=None, plot=False):
    loader.chunk_length = chunk     # re-split BOLD
    loader._split_into_chunks()

    sc = loader.get_subject_connectome(subject, norm=True) # log normalized
    sim = RWWSubjectSimulator(sc, node_size=sc.shape[0], step_size=step, TP_per_window=chunk, g_init=g, use_fic=True)
    sim.model.to(DEVICE)

    emp_FCs = [fc for _, fc in loader.iter_chunks_per_subject(subject)]

    cost = CostsRWW(sim.model, use_rate_reg=args.rate_reg, lambda_rate=args.lambda_rate,
                               use_spec_reg=args.spec_reg, lambda_spec=args.lambda_spec,
                               use_disp_reg=args.disp_reg, lambda_disp=args.lambda_disp)

    fit = Model_fitting(sim.model, cost, device=DEVICE)
    fit.train(u=0, empFcs=emp_FCs, num_epochs=epochs, num_windows=1, learningrate=lr, max_chunks=20)

    emp_ts = loader.all_bold[subject]
    ts_len = emp_ts.shape[1]
    num_windows = ts_len // sim.model.TRs_per_window
    ts_sim, fc_sim = fit.simulate(u=0, num_windows=num_windows)

    emp_FC = torch.tensor(np.corrcoef(loader.all_bold[subject]), dtype=torch.float32, device=DEVICE)
    mask = np.tril_indices(emp_FC.shape[0], -1)
    corr = np.corrcoef(fc_sim[mask], emp_FC.cpu().numpy()[mask])[0, 1]

    print(f"Subject {subject:02d}  |  FC-corr = {corr:.3f}")

    if plot:
        outdir = f"{g}_{step}_{chunk}"
        heatmap_fc(fc_sim, emp_FC.cpu().numpy(), subj=subject, r=corr, subdir=outdir)
        scatter_fc(fc_sim, emp_FC, subj=subject, r=corr, subdir=outdir)
        timeseries(ts_sim[:, 10:], emp_ts, tr=sim.model.tr, subdir=outdir)
        sc_orig = sim.sc.cpu().numpy()
        sc_fit = sim.model.sc_fitted.detach().cpu().numpy()
        heatmap_sc(sc_orig, sc_fit, subj=subject, tag="sc_gain", subdir=outdir)

        delta         = sc_fit - sc_orig
        decreased_idx = delta < 0
        neg_final_idx = sc_fit < 0
        print(f"[SC] {decreased_idx.sum()} / {delta.size} connections decreased (median Δ = {np.median(delta[decreased_idx]):.4f})")
        print(f"[SC] {neg_final_idx.sum()} connections are negative after training")



    return corr

def main(args):
    fmri_filename = "/vol/bitbucket/ank121/fyp/HCP Data/BOLD Timeseries HCP.mat"
    sc_dir = "/vol/bitbucket/ank121/fyp/HCP Data/distance_matrices/"
    distance_matrix_path = "/vol/bitbucket/ank121/fyp/HCP Data/schaefer100_dist.npy"

    loader = BOLDDataLoader(fmri_filename, sc_dir, distance_matrix_path, chunk_length=50)

    grid = list(itertools.product(args.g, args.step, args.chunk))
    results = {}
    for g, step, chunk in grid:
        subject = args.subject if args.subject else np.random.randint(low=1, high=loader.num_subjects)
        print(f"[Subject {subject}\n\tg = {g},\n\tstep = {step*1000}ms,\n\tchunk size = {chunk} TRs,\n\tlr = {args.lr}\n]")
        score = run_single(loader, subject, g, step, chunk, epochs=args.epochs, lr=args.lr, args=args, plot=True)
        results[(g, step, chunk)] = score
        print(f">> Mean FC-corr = {score:.3f}")

    # save JSON
    with open("grid_results.json","w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)

    best_cfg, best_score = max(results.items(), key=lambda kv: kv[1])
    print(f"Best cfg  g={best_cfg[0]}  step={best_cfg[1]}  chunk={best_cfg[2]} -> r={best_score:.3f}")

    run_single(loader, 42, *best_cfg, epochs=args.epochs, lr=args.lr, args=args, plot=False)


if __name__ == "__main__":
    print("[MAIN] Starting grid search..")
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default = 35)
    p.add_argument("--lr",     type=float, default = 0.05) # 0.05
    p.add_argument("--subject", type=int)
    p.add_argument("--g",      nargs="+", type=float, default = [20, 50, 80, 100, 500, 1000])
    p.add_argument("--step",   nargs="+", type=float, default = [0.01, 0.05, 0.1])
    p.add_argument("--chunk",  nargs="+", type=int, default = [30, 50, 100])
    p.add_argument("--rate_reg", action="store_true", help="activate firing-rate regulariser")
    p.add_argument("--spec_reg", action="store_true")
    p.add_argument("--disp_reg", action="store_true")
    p.add_argument("--lambda_rate", type=float, default=0.1)
    p.add_argument("--lambda_spec", type=float, default=0.05)
    p.add_argument("--lambda_disp", type=float, default=0.05)


    args = p.parse_args()
    main(args)
