""" Generate synthetic 5-node 'ground-truth' data """

import numpy as np, torch, json, time
import matplotlib.pyplot as plt
from pathlib import Path
from whobpyt.data_loader import DEVICE
from simulators.rww_simulator import RWWSubjectSimulator
from whobpyt.custom_cost_RWW import CostsRWW
from whobpyt.modelfitting import Model_fitting
from whobpyt.utils.plotter import heatmap_fc, timeseries
import seaborn as sns

def create_random_ring(n_nodes=5, seed=None):
    """Create a ring network with random edge weights."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    ring = torch.zeros(n_nodes, n_nodes)
    for i in range(n_nodes):
        ring[i, (i + 1) % n_nodes] = torch.randn(1).item()  
    return ring

def plot_ring_network(ring, save_path):
    """ Plot the ring network with edge weights """
    plt.figure(figsize=(8, 6))
    sns.heatmap(ring.cpu().numpy(), 
                cmap='coolwarm',
                center=0,
                square=True,
                cbar_kws={'label': 'Edge Weight'})
    plt.title('Ring Network Structure')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

SAVE_DIR = Path("exp1_outputs"); SAVE_DIR.mkdir(parents=True, exist_ok=True)

N_NODES     = 5
g_star      = 500.0         # long-range gain
g_EE_star   = 3.5           # E -> E
g_EI_star   = 0.42          # E -> I
SEED        = None

STD_IN      = 0.0
BASE_WINDOW = 10            # warmup - throw away
STEP        = 0.05          # s
TP_PER_WIN  = 50
N_WINDOWS   = 80            # 50·24·0.05 s = 60 s BOLD 

# Create and visualize ring network
ring = create_random_ring(n_nodes=N_NODES, seed=SEED)  # Using seed for reproducibility
ring = ring.to(dtype=torch.float32, device=DEVICE)
plot_ring_network(ring, SAVE_DIR / "ring_graph.png")
np.save(SAVE_DIR / "ring_sc.npy", ring.cpu().numpy())


for N_WINDOWS in [80]:
    sim = RWWSubjectSimulator(ring, node_size=N_NODES, g_init=g_star, step_size=STEP,
                            TP_per_window=TP_PER_WIN, use_fic=False, std_in=STD_IN)
    sim.model.to(DEVICE)
    # overwrite local gains
    sim.model.params.g_EE.val.data = torch.tensor(g_EE_star, device=DEVICE)
    sim.model.params.g_EI.val.data = torch.tensor(g_EI_star, device=DEVICE)

    cost_dummy = CostsRWW(sim.model)  # unused
    fit_dummy  = Model_fitting(sim.model, cost_dummy, device=DEVICE)
    ts_gt, fc_gt = sim.simulate(u=0, num_windows=N_WINDOWS, base_window_num=BASE_WINDOW, transient_num=10)
    
    sim2 = RWWSubjectSimulator(ring, node_size=N_NODES, g_init=g_star, step_size=STEP,
                            TP_per_window=TP_PER_WIN, use_fic=False, std_in=STD_IN)
    sim2.model.to(DEVICE)
    # overwrite local gains
    sim2.model.params.g_EE.val.data = torch.tensor(g_EE_star, device=DEVICE)
    sim2.model.params.g_EI.val.data = torch.tensor(g_EI_star, device=DEVICE)

    cost_dummy = CostsRWW(sim2.model)  # unused
    fit_dummy  = Model_fitting(sim2.model, cost_dummy, device=DEVICE)
    # Test 1: Same parameters
    ts_gt2, fc_gt2 = sim2.simulate(u=0, num_windows=N_WINDOWS, base_window_num=BASE_WINDOW, transient_num=10)

    sim3 = RWWSubjectSimulator(ring, node_size=N_NODES, g_init=g_star, step_size=STEP,
                            TP_per_window=TP_PER_WIN, use_fic=False, std_in=STD_IN)
    sim3.model.to(DEVICE)
    # overwrite local gains
    sim3.model.params.g.val.data = torch.tensor(g_star * 1.1, device=DEVICE)
    sim3.model.params.g_EE.val.data = torch.tensor(g_EE_star * 1.1, device=DEVICE)

    cost_dummy = CostsRWW(sim3.model)  # unused
    fit_dummy  = Model_fitting(sim3.model, cost_dummy, device=DEVICE)
    # Test 2: 10% different parameters
    ts_gt3, fc_gt3 = sim3.simulate(u=0, num_windows=N_WINDOWS, base_window_num=BASE_WINDOW, transient_num=10)

    # Compute correlations
    corr_12 = np.corrcoef(fc_gt.flatten(), fc_gt2.flatten())[0,1]
    corr_13 = np.corrcoef(fc_gt.flatten(), fc_gt3.flatten())[0,1]
    corr_23 = np.corrcoef(fc_gt2.flatten(), fc_gt3.flatten())[0,1]

    print(f"WINDOW SIZE = {N_WINDOWS}:")
    print(f"Original vs Same params: {corr_12:.3f}")
    print(f"Original vs +10% params: {corr_13:.3f}")
    print(f"Same vs +10% params: {corr_23:.3f}")

# np.save(SAVE_DIR / "gt_ts.npy", ts_gt)
# np.save(SAVE_DIR / "gt_fc.npy", fc_gt)
# meta = dict(datetime=time.ctime(), g=g_star, g_EE=g_EE_star, g_EI=g_EI_star, step=STEP, TP_per_window=TP_PER_WIN, n_windows=N_WINDOWS)
# json.dump(meta, open(SAVE_DIR / "meta.json", "w"), indent=2)

# # ground-truth FC
# heatmap_fc(fc_gt, r=None, tag="fc", subdir="synthetic", outdir="exp1_outputs")

# # ground-truth TS (first 400 TRs)
# plt.figure(figsize=(6,3)); plt.plot(ts_gt[:, :400].T)
# plt.title("Ground-truth BOLD (first 400 TR)"); plt.xlabel("TR"); plt.ylabel("Signal (a.u.)")
# plt.tight_layout(); plt.savefig(SAVE_DIR/"gt_timeseries.png", dpi=300); plt.close()

# ts_gt_1 = ts_gt[:, :400]
# ts_gt_2 = ts_gt[:, 300:700]
# timeseries(ts_gt_2, ts_gt_1, max_nodes=5, tag="timeseries", subdir="synthetic", outdir="exp1_outputs")


# print("[INFO] ground-truth written to", SAVE_DIR.resolve())
