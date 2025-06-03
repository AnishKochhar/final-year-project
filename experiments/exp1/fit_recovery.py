import argparse, json, time
from pathlib import Path
import numpy as np, torch
import matplotlib.pyplot as plt
from whobpyt.data_loader import DEVICE
from simulators.rww_simulator import RWWSubjectSimulator
from whobpyt.custom_cost_RWW import CostsRWW
from whobpyt.modelfitting import Model_fitting

SAVE_DIR = Path("exp1_outputs")
GT   = np.load(SAVE_DIR / "gt_fc.npy")
SCnp = np.load(SAVE_DIR / "ring_sc.npy")
SC   = torch.tensor(SCnp, dtype=torch.float32, device=DEVICE)


parser = argparse.ArgumentParser()
parser.add_argument("--runs",    type=int, default=10, help="independent optimiser restarts")
parser.add_argument("--epochs",  type=int, default=20, help="max training epochs")
parser.add_argument("--windows", type=int, default=8,  help="windows drawn per epoch")
args = parser.parse_args()

results = []          


for run in range(args.runs):
    torch.manual_seed(run)
    np.random.seed(run)

    # Random initial guesses (wide but positive)
    g0     = float(np.random.uniform(10, 150))
    gEE0   = float(np.random.uniform(1.0, 6.0))
    print(f"[RUN {run:02d}] init g={g0:5.1f}  gEE={gEE0:4.2f}")

    sim = RWWSubjectSimulator(SC, node_size=5, g_init=g0,
                              step_size=0.05, TP_per_window=50,
                              use_fic=False)
    sim.model.to(DEVICE)
    print(sim.model.params_fitted.keys())
    for n,p in sim.model.params_fitted.items():
        print(n, p.shape)

    break

    sim.model.params.g_EE.val.data = torch.tensor(gEE0, device=DEVICE)

    cost = CostsRWW(sim.model)           # FC-corr only
    fitter = Model_fitting(sim.model, cost, device=DEVICE)

    fitter.train(u=0,
                 empFcs=[torch.tensor(GT, dtype=torch.float32, device=DEVICE)],
                 num_epochs=args.epochs,
                 num_windows=args.windows,
                 learningrate=0.05,
                 lr_2ndLevel=0.02,
                 lr_scheduler=False,
                 early_stopping=True,
                 max_chunks=1)            # one FC per epoch, randomly sampled

    g_est   = sim.model.params.g.val.item()
    gEE_est = sim.model.params.g_EE.val.item()

    _, fc_sim = fitter.simulate(u=0, num_windows=8, transient_num=10)
    fc_corr   = np.corrcoef(fc_sim[np.tril_indices(5, -1)],
                            GT[np.tril_indices(5, -1)])[0, 1]
    
    print(f"[g={g_est:.3f} gEE={gEE_est:.3f}] FC-corr={fc_corr:.4f}")
    results.append(dict(run=run, g0=g0, gEE0=gEE0,
                        g_est=g_est, gEE_est=gEE_est,
                        fc=fc_corr))

# json.dump(results, open(SAVE_DIR / "recover_results.json", "w"), indent=2)

# g_err   = [r["g_est"]   - 80.0 for r in results]
# gEE_err = [r["gEE_est"] - 3.5  for r in results]

# plt.figure(figsize=(6, 4))
# plt.boxplot([g_err, gEE_err], labels=[r"$g$", r"$g_{EE}$"])
# plt.axhline(0, ls="--", c="k", lw=.5)
# plt.ylabel("Estimation error")
# plt.title("Parameter-recovery errors across restarts")
# plt.tight_layout()
# plt.savefig(SAVE_DIR / "recover_errors.png", dpi=300)
# plt.close()

# print("[DONE] wrote recover_results.json + recover_errors.png")
