""" Generate synthetic 5-node 'ground-truth' data """

import numpy as np, torch, json, time
from pathlib import Path
from whobpyt.data_loader import DEVICE
from simulators.rww_simulator import RWWSubjectSimulator
from whobpyt.custom_cost_RWW import CostsRWW
from whobpyt.modelfitting import Model_fitting

SAVE_DIR = Path("exp1_outputs"); SAVE_DIR.mkdir(parents=True, exist_ok=True)

g_star      = 80.0          # long-range gain
g_EE_star   = 3.5           # E -> E
g_EI_star   = 0.42          # E -> I
STEP        = 0.05          # ms
TP_PER_WIN  = 50
N_WINDOWS   = 800           # 50·800·0.05 ms = 2 s BOLD per window → 26 min

ring = torch.zeros(5, 5)
for i in range(5):
    ring[i, (i + 1) % 5] = 1
ring = ring.to(dtype=torch.float32, device=DEVICE)
np.save(SAVE_DIR / "ring_sc.npy", ring.cpu().numpy())

sim = RWWSubjectSimulator(ring, node_size=5, g_init=g_star, step_size=STEP,
                          TP_per_window=TP_PER_WIN, use_fic=False)
sim.model.to(DEVICE)
# overwrite local gains
sim.model.params.g_EE.val.data = torch.tensor(g_EE_star, device=DEVICE)
sim.model.params.g_EI.val.data = torch.tensor(g_EI_star, device=DEVICE)


cost_dummy = CostsRWW(sim.model)  # unused
fit_dummy  = Model_fitting(sim.model, cost_dummy, device=DEVICE)
ts_gt, fc_gt = fit_dummy.simulate(u=0, num_windows=N_WINDOWS)

np.save(SAVE_DIR / "gt_ts.npy", ts_gt)
np.save(SAVE_DIR / "gt_fc.npy", fc_gt)
meta = dict(datetime=time.ctime(), g=g_star, g_EE=g_EE_star, g_EI=g_EI_star,
            step=STEP, TP_per_window=TP_PER_WIN, n_windows=N_WINDOWS)
json.dump(meta, open(SAVE_DIR / "meta.json", "w"), indent=2)
print("[INFO] ground-truth written to", SAVE_DIR.resolve())
