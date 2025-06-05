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

SAVE_DIR = Path("exp1_outputs"); SAVE_DIR.mkdir(parents=True, exist_ok=True)

N_NODES     = 8
G_star      = 500.0         # long-range gain
G_EE_star   = 3.5           # E -> E
G_EI_star   = 0.42          # E -> I
G_IE_star   = 0.42
SEED        = None

STD_IN      = 0.0
BASE_WINDOW = 10            # warmup - throw away
STEP        = 0.05          # s
TP_PER_WIN  = 50
N_WINDOWS   = 200            # 50·200·0.05 s = 500 s BOLD 
K_FC = 10

if SEED is not None:
    torch.manual_seed(SEED); np.random.seed(SEED)
ring = torch.zeros(N_NODES, N_NODES)
for i in range(N_NODES):
    ring[i, (i+1) % N_NODES] = np.random.randn()        # random sign / weight
# normalise to ||SC||_F = 1  (same as BOLDDataLoader)
ring = ring / torch.norm(ring, p='fro')
np.save(SAVE_DIR / "ring_sc.npy", ring.cpu().numpy())

plt.figure(figsize=(4,3))
sns.heatmap(ring.cpu(), cmap='coolwarm', center=0, square=True, cbar=False)
plt.title("Normalised ring SC"); plt.tight_layout()
plt.savefig(SAVE_DIR/"ring_sc.png", dpi=200); plt.close()

sim = RWWSubjectSimulator(ring, node_size=N_NODES, g_init=G_star, g_IE_init=G_IE_star, 
                            g_EI_init=G_EI_star, g_EE_init=G_EE_star, step_size=STEP,
                            TP_per_window=TP_PER_WIN, use_fic=False, std_in=STD_IN)
sim.model.to(DEVICE)

ts_gt, _ = sim.simulate(u=0, num_windows=N_WINDOWS,
                        base_window_num=BASE_WINDOW, transient_num=10)  
np.save(SAVE_DIR/"gt_ts.npy", ts_gt)

fc_list = []
for k in range(K_FC):
    start = k * TP_PER_WIN * (N_WINDOWS // K_FC)   # non-overlap blocks
    stop  = start + TP_PER_WIN * (N_WINDOWS // K_FC)
    fc_k  = np.corrcoef(ts_gt[:, start:stop])
    # heatmap_fc(fc_k, tag=f"fc_{k}", outdir="exp1_outputs", subdir="synthetic")
    fc_list.append(fc_k)
np.save(SAVE_DIR/"gt_fc_list.npy", np.stack(fc_list))

grand_fc = np.corrcoef(ts_gt)
np.save(SAVE_DIR/"gt_fc.npy", grand_fc)

meta = dict(datetime=time.ctime(), g=G_star, g_EE=G_EE_star, g_EI=G_EI_star, g_IE=G_IE_star,
            n_nodes=N_NODES, n_windows=N_WINDOWS, tp_per_win=TP_PER_WIN)
json.dump(meta, open(SAVE_DIR/"meta.json","w"), indent=2)
print("[generate_synthetic] Wrote normalised SC, gt_ts.npy, gt_fc_list.npy")
