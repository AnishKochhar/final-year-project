""" Experiment 4 - vector FIC  vs scalar g_IE """

import argparse, json, os, numpy as np, torch, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path

from whobpyt.data_loader import BOLDDataLoader, DEVICE
from simulators.rww_simulator import RWWSubjectSimulator
from whobpyt.custom_cost_RWW import CostsRWW
from whobpyt.modelfitting import Model_fitting

def fit_subject(loader, subject_id, use_fic, epochs, lr):
    sc = loader.get_subject_connectome(subject_id, norm=True)
    sim = RWWSubjectSimulator(sc, node_size=sc.shape[0], use_fic=use_fic, g_init=50, step_size=0.05)
    sim.model.to(DEVICE)

    emp_FCs = [fc for _, fc in loader.iter_chunks_per_subject(subject_id)]
    cost = CostsRWW(sim.model)
    fit = Model_fitting(sim.model, cost, device=DEVICE)
    fit.train(u=0, empFcs=emp_FCs, num_epochs=epochs, num_windows=1, learningrate=lr, early_stopping=True)

    n_epochs = len(fit.trainingStats.loss) # in case of early stopping
    emp_ts = loader.all_bold[subject_id]
    n_win = emp_ts.shape[1] // sim.model.TRs_per_window
    _, fc_s = fit.simulate(u=0, num_windows=n_win)
    fc_e = np.corrcoef(emp_ts)
    m = np.tril_indices(fc_e.shape[0], -1)
    corr = np.corrcoef(fc_s[m], fc_e[m])[0,1]

    return corr, n_epochs

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="/vol/bitbucket/ank121/fyp/HCP Data")
    p.add_argument("--subjects",  nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--epochs",    type=int,  default=30)
    p.add_argument("--lr",        type=float, default=0.05)
    args = p.parse_args()


    distance_matrices_path = os.path.join(args.data_root, "schaefer100_dist.npy")
    fmri_path = os.path.join(args.data_root, "BOLD Timeseries HCP.mat")
    sc_dir = os.path.join(args.data_root, "distance_matrices")
    loader = BOLDDataLoader(fmri_path, sc_dir, distance_matrices_path, chunk_length=50)
    loader._split_into_chunks()

    stats = {"fic": [], "scalar": []}
    for subject in args.subjects:
        corr, ep = fit_subject(loader, subject, True, args.epochs, args.lr)
        stats["fic"].append((corr, ep))
        corr, ep = fit_subject(loader, subject, False, args.epochs, args.lr)
        stats["scalar"].append((corr, ep))

    out = Path("exp4_outputs"); out.mkdir(exist_ok=True)

    with open(out / "results.json","w") as fp:
        json.dump({k:[float(x) for x in v] for k,v in stats.items()}, fp, indent=2)


    def make_plot(metric_idx, ylabel, fname):
        vals = {"FIC": [s[metric_idx] for s in stats["fic"]],
                "Scalar": [s[metric_idx] for s in stats["scalar"]]}
        sns.boxplot(data=vals); sns.swarmplot(data=vals, color=".3", size=6)
        plt.ylabel(ylabel); plt.tight_layout()
        plt.savefig(out / fname, dpi=300); plt.close()
    
    make_plot(0, "FC correlation", "fc_corr_box.png")
    make_plot(1, "Epochs to convergence", "epochs_box.png")


if __name__ == "__main__":
    main()