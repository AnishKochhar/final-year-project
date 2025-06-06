""" Simulate and save full-length BOLD """

import argparse, json, os, numpy as np, torch
from pathlib import Path
import matplotlib.pyplot as plt

from whobpyt.data_loader import BOLDDataLoader, DEVICE
from simulators.rww_simulator import RWWSubjectSimulator
from whobpyt.custom_cost_RWW import CostsRWW
from whobpyt.modelfitting import Model_fitting
from whobpyt.utils.plot_fc_windows import plot_fc_grid
from whobpyt.utils.plotter import heatmap_fc   

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

    subj = args.subj
    ts_test, fc_test = loader.test_fc_and_ts(subj, win_len=200)  # numpy, torch
    fc_test_np = fc_test.cpu().numpy()

    grid_chunks = [100, 50, 20]
    grid_flags  = [(e, i) for e in [True, False] for i in [True, False]]
    results = {}

    for chunk in grid_chunks:
        loader.chunk_length = chunk 
        loader._split_into_chunks()
        for fit_g_EE, fit_g_EI in grid_flags:
            label = f"sub{subj:03d}_chunk{chunk}_gEE{int(fit_g_EE)}_gEI{int(fit_g_EI)}"
            print(f"[+] {label}")

            sc = loader.get_subject_connectome(subj, norm=True)
            sim = RWWSubjectSimulator(sc, sc.shape[0],
                                    TP_per_window=chunk, fit_g_EE=fit_g_EE, fit_g_IE=True, fit_g_EI=fit_g_EI, 
                                    use_bifurcation=False, step_size=args.step, g_init=args.g, use_fic=False)
            sim.model.to(DEVICE)

            # emp_FCs = [fc for _, fc in loader.iter_chunks_per_subject(subj)]
            # full_FC = torch.as_tensor(np.corrcoef(loader.all_bold[subj]), dtype=torch.float32, device=DEVICE)
            # emp_FCs = [full_FC]
            emp_FCs = loader.train_fc_windows(subj, win_len=200)
            # num_windows = loader.all_bold[subj].shape[1] // chunk
            cost = CostsRWW(sim.model,
                            use_rate_reg = args.lambda_rate > 0, lambda_rate = args.lambda_rate,
                            use_spec_reg = args.lambda_spec > 0, lambda_spec = args.lambda_spec,
                            use_disp_reg = args.lambda_disp > 0, lambda_disp = args.lambda_disp)
            fitter = Model_fitting(sim.model, cost, device=DEVICE)
            fitter.train(u=0, empFcs=emp_FCs, num_epochs=args.epochs,
                        num_windows=len(emp_FCs), learningrate=args.lr, early_stopping=True)

            chunk = sim.model.TRs_per_window
            n_win = emp_FCs[0].shape[1] // chunk
            ts_sim, fc_sim = sim.simulate(u=0, num_windows=n_win, base_window_num=10)
            tri = np.tril_indices_from(fc_test_np, k=-1)
            r = np.corrcoef(fc_sim[tri], fc_test_np[tri])[0, 1]

            results[label] = float(r)
            print(f"- FC-corr test = {r:.3f} -")

            # Save heat-map
            heatmap_fc(fc_sim, fc_test, subj=subj, r=r, subtitle=label, tag=label, subdir="sim_fc")

    Path("grid_results").mkdir(exist_ok=True)
    with open("grid_results/summary.json", "w") as fp:
        json.dump(results, fp, indent=2)
    print("\nFinished. Summary written to grid_results/summary.json")

if __name__ == "__main__":
    torch.manual_seed(0); np.random.seed(0)
    main()
