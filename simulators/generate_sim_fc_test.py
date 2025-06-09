""" Simulate and save full-length BOLD """

import argparse, json, os, numpy as np, torch
from pathlib import Path
import matplotlib.pyplot as plt
from pathlib import Path

from whobpyt.data_loader import BOLDDataLoader, DEVICE
from simulators.rww_simulator import RWWSubjectSimulator
from whobpyt.custom_cost_RWW import CostsRWW
from whobpyt.modelfitting import Model_fitting
from whobpyt.utils.fc_tools import bold_to_fc
from whobpyt.utils.plot_fc_windows import plot_fc_grid
from whobpyt.utils.plotter import heatmap_fc, timeseries

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/vol/bitbucket/ank121/fyp/HCP Data")
    p.add_argument("--subj", type=int, default=18)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--g", type=float, default=1000)
    p.add_argument("--step", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--lambda-rate", type=float, default=0.05)
    p.add_argument("--lambda-spec", type=float, default=0.05)
    p.add_argument("--lambda-disp", type=float, default=0.03)
    args = p.parse_args()

    fmri = os.path.join(args.data_root, "BOLD Timeseries HCP.mat")
    scdir = os.path.join(args.data_root, "distance_matrices")
    dist  = os.path.join(args.data_root, "schaefer100_dist.npy")

    # Load once (no chunk yet)
    loader = BOLDDataLoader(fmri, scdir, dist)

    SAVE_DIR = Path("simulated_outputs"); SAVE_DIR.mkdir(parents=True, exist_ok=True)

    subj = args.subj

    # grid_chunks = [50, 100]
    # gs = [800, 1000, 1200]
    # grid_flags  = [(e, i, fic) for e in [True, False] for i in [True, False] for fic in [True, False]]
    grid_chunks = [100]
    gs = [1200]
    grid_flags  = [(e, i, fic) for e in [False] for i in [True] for fic in [True]]
    results = {}
    
    ts_emp = loader.all_bold[subj]
    fc_emp = bold_to_fc(ts_emp)

    for chunk in grid_chunks:
        loader.chunk_length = chunk 
        loader._split_into_chunks()
        for g in gs:
            for fit_g_EE, fit_g_EI, use_fic in grid_flags:
                label = f"sub{subj:03d}_g{g}_chunk{chunk}_gEE{int(fit_g_EE)}_gEI{int(fit_g_EI)}_fic{int(use_fic)}"
                print(f"[+] {label}")

                sc = loader.get_subject_connectome(subj, norm=True)
                sim = RWWSubjectSimulator(sc, sc.shape[0],
                                        TP_per_window=chunk, fit_g_EE=fit_g_EE, fit_g_IE=not use_fic, fit_g_EI=fit_g_EI, 
                                        use_bifurcation=False, step_size=args.step, g_init=g, use_fic=use_fic)
                sim.model.to(DEVICE)

                # emp_FCs = [fc for _, fc in loader.iter_chunks_per_subject(subj)]
                # full_FC = torch.as_tensor(np.corrcoef(loader.all_bold[subj]), dtype=torch.float32, device=DEVICE)
                # emp_FCs = [full_FC]            
                train_window_length = 900
                emp_FCs = loader.train_fc_windows(subj, win_len=train_window_length)
                train_num_windows = 2000 // chunk

                cost = CostsRWW(sim.model,
                                use_rate_reg = args.lambda_rate > 0, lambda_rate = args.lambda_rate,
                                use_spec_reg = args.lambda_spec > 0, lambda_spec = args.lambda_spec,
                                use_disp_reg = args.lambda_disp > 0, lambda_disp = args.lambda_disp)
                fitter = Model_fitting(sim.model, cost, device=DEVICE)
                fitter.train(u=0, empFcs=emp_FCs, num_epochs=args.epochs,
                            num_windows=train_num_windows, learningrate=args.lr, early_stopping=True)

                test_num_windows = ts_emp.shape[1] // chunk
                corrs = []
                for _ in range(3):
                    ts_sim, fc_sim = sim.simulate(u=0, num_windows=test_num_windows, base_window_num=10)
                    tri = np.tril_indices_from(fc_emp, k=-1)
                    r = np.corrcoef(fc_sim[tri], fc_emp[tri])[0, 1]
                    corrs.append(r)
                    print(f"r = {r}")

                mean_corr = np.mean(corrs)
                std_corr = np.std(corrs)
                print(f"- Entire TS FC-corr test = {mean_corr:.3f}, std = {std_corr:.3f} -")

                # Save heat-map
                # heatmap_fc(fc_sim, fc_test, subj=subj, r=r, subtitle=label, tag=label, subdir="sim_fc")
                if mean_corr > 0.5:
                    np.save(SAVE_DIR / f"{label}.npy", ts_sim)
                    timeseries(ts_sim[:, 50:100], ts_emp[:, 50:100], max_nodes=6, tr=0.75, sep=True, tag=label, subdir="plots", outdir="simulated_outputs")

    Path("grid_results").mkdir(exist_ok=True)
    with open("grid_results/summary.json", "w") as fp:
        json.dump(results, fp, indent=2)
    print("\nFinished. Summary written to grid_results/summary.json")

if __name__ == "__main__":
    torch.manual_seed(0); np.random.seed(0)
    main()
