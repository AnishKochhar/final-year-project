import argparse, json, time, itertools
from pathlib import Path
import numpy as np, torch
import matplotlib.pyplot as plt, seaborn as sns

from whobpyt.data_loader import DEVICE
from simulators.rww_simulator import RWWSubjectSimulator
from whobpyt.custom_cost_RWW import CostsRWW
from whobpyt.modelfitting import Model_fitting
from whobpyt.utils.plotter import heatmap_fc

def sliding_dfc(ts, win=50, stride=25):
    """ Return (W,E) torch tensor of FC vectors """
    if isinstance(ts, np.ndarray):
        ts = torch.tensor(ts, device=DEVICE)
    N, T = ts.shape
    tri = torch.tril_indices(N, N, -1)
    vecs = []
    for s in range(0, T - win + 1, stride):
        fc = torch.corrcoef(ts[:, s:s+win])[tri[0], tri[1]]
        vecs.append(fc)
    return torch.stack(vecs)           

@torch.no_grad()
def dfc_similarity(ts_a, ts_b, win=50, stride=25):
    """ Pearson corr between windowed FC trajectories """
    A = sliding_dfc(ts_a, win, stride)
    B = sliding_dfc(ts_b, win, stride)
    corrs = [torch.corrcoef(torch.stack([a, b]))[0, 1] for a, b in zip(A, B)]
    print(corrs)
    return torch.stack(corrs).mean().item()

@torch.no_grad()
def spectral_beta(ts, fs=1/0.75, f_lo=0.02, f_hi=0.1):
    """ Return PSD slope beta (positive => 1/f^beta) """
    ts = ts - ts.mean(1, keepdims=True)
    fft  = torch.fft.rfft(ts, dim=1)
    psd  = (fft.real**2 + fft.imag**2).mean(0)        # (F,)
    freqs = torch.fft.rfftfreq(ts.shape[1], d=1/fs).to(ts.device)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    x = torch.log(freqs[mask] + 1e-8); y = torch.log(psd[mask] + 1e-8)
    beta = -torch.dot(x - x.mean(), y - y.mean()) / torch.dot(x - x.mean(), x - x.mean())
    return beta.item()



SAVE_DIR = Path("exp1_outputs")
GT_FC     = np.load(SAVE_DIR / "gt_fc.npy")                 # grand FC
GT_TS     = np.load(SAVE_DIR / "gt_ts.npy")                 # (N, T)
FC_LIST   = np.load(SAVE_DIR / "gt_fc_list.npy")            # (K, N, N)
empFcs    = [torch.tensor(fc, dtype=torch.float32, device=DEVICE) for fc in FC_LIST]
SC        = torch.tensor(np.load(SAVE_DIR / "ring_sc.npy"), dtype=torch.float32, device=DEVICE)


N_NODES   = SC.shape[0]
TP_PER_WIN = 50
STEP = 0.05
BASE_WINDOW = 10    # burn-in windows
TRAIN_WINDOWS = 8   # mini-batch in time
EVAL_WINDOWS = 80   # 200 s per evaluation

# anchor start-values for sparse grid (six points)
ANCHOR_G  = [200, 300, 400, 500, 600, 700]
G_EE_TRUE = 3.5
G_EI_TRUE = 0.42
G_IE_TRUE = 0.42
              

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=20)
args = parser.parse_args()

# store trajectories for plotting
traj = {g_start: [] for g_start in ANCHOR_G}
metrics = []

for run, g0 in enumerate(ANCHOR_G):
    # torch.manual_seed(42); np.random.seed(42)
    print(f"\n[RUN {run:02d}] start g={g0}")

    sim = RWWSubjectSimulator(SC, node_size=N_NODES,
                              g_init=g0, g_EE_init=G_EE_TRUE,
                              g_EI_init=G_EI_TRUE, g_IE_init=G_IE_TRUE,
                              fit_g_EE=False, fit_g_EI=False, fit_g_IE=False, 
                              fit_gains=False, use_bifurcation=False,
                              step_size=STEP, TP_per_window=TP_PER_WIN,
                              use_fic=False, std_in=0.0)
    sim.model.to(DEVICE)

    cost = CostsRWW(sim.model, use_rate_reg=True, use_disp_reg=True, use_spec_reg=True, log_loss=True)
    fitter = Model_fitting(sim.model, cost, device=DEVICE)

    fitter.train(u=0,
                 empFcs=empFcs, max_chunks=3,        # sample 3 FC matrices
                 num_epochs=args.epochs, num_windows=TRAIN_WINDOWS,
                 learningrate=0.3, lr_2ndLevel=0.02,
                 lr_scheduler=True, early_stopping=False)            

    g_est = sim.model.params.g.val.item()
    traj[g0].extend(fitter.trainingStats.fit_params["g"])

    ts_sim, fc_sim = fitter.simulate(u=0, num_windows=EVAL_WINDOWS,
                                     base_window_num=BASE_WINDOW, transient_num=10)

    # static FC corr
    r_static = np.corrcoef(fc_sim[np.tril_indices(N_NODES, -1)],
                           GT_FC[np.tril_indices(N_NODES, -1)])[0,1]

    # dynamic-FC corr
    r_dfc = dfc_similarity(ts_sim, GT_TS)

    # spectral exponent error
    beta_sim = spectral_beta(torch.tensor(ts_sim, device=DEVICE))
    beta_gt  = spectral_beta(torch.tensor(GT_TS,  device=DEVICE))
    err_beta = abs(beta_sim - beta_gt)

    metrics.append(dict(g0=g0, g_est=g_est,
                        r_static=r_static, r_dfc=r_dfc,
                        beta_sim=beta_sim, beta_gt=beta_gt, err_beta=err_beta))

    print(f" final g={g_est:.1f}   r_static={r_static:.3f}   "
          f"r_dFC={r_dfc:.3f}   Δβ={err_beta:.3f}")
    
json.dump(metrics, open(SAVE_DIR/"recover_results.json","w"), indent=2)

    
plt.figure(figsize=(5,3))
for g0, series in traj.items():
    plt.plot(series, label=f"g₀={g0}")
plt.xlabel("window update"); plt.ylabel("g value")
plt.legend(fontsize=7); plt.tight_layout()
plt.savefig(SAVE_DIR/"recovery_traj.png", dpi=200); plt.close()

plt.figure(figsize=(4,3))
for m in metrics:
    plt.scatter(m["r_static"], m["r_dfc"],
                c=np.sign(m["g_est"]-500), cmap="coolwarm", s=40)
plt.xlabel("static FC corr"); plt.ylabel("mean dFC corr")
plt.tight_layout(); plt.savefig(SAVE_DIR/"recovery_metrics.png", dpi=200); plt.close()
print("[DONE] wrote recover_results.json + plots")


