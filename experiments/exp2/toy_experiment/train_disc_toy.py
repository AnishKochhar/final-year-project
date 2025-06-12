import argparse, json, time, random
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from whobpyt.data_loader import DEVICE
from simulators.rww_simulator import RWWSubjectSimulator
from whobpyt.custom_cost_RWW import CostsRWW
from whobpyt.models.fc_cnn_disc import FCCNNDisc

def make_ring_sc(n=5, noise_std=0.05):
    """ Return ring SC with optional gaussian weight noise """
    sc = np.eye(n, k=1) + np.eye(n, k=-1)
    sc[0, -1] = sc[-1, 0] = 1
    if noise_std > 0:
        sc = sc + np.random.normal(scale=noise_std, size=sc.shape)
        sc[sc < 0] = 0
    return sc / sc.sum(axis=1, keepdims=True)


def simulate_fc(sim, chunk_len, n_chunks, base=10):
    """ Run a simulator long enough, cut into chunks, return (B, n, n) """
    usable_ts = n_chunks * chunk_len
    ts, _ = sim.simulate(u=0, num_windows=usable_ts // chunk_len,
                         base_window_num=base)      # (n, T)
    idxs = np.arange(0, ts.shape[1] - chunk_len + 1, chunk_len)
    fcs = [np.corrcoef(ts[:, i:i+chunk_len]) for i in idxs[:n_chunks]]
    return np.stack(fcs, axis=0)                    # (B, n, n)


class FCDataset(Dataset):
    """ Holds FC matrices (real=1 or fake=0) """
    def __init__(self, real_fcs, fake_fcs):
        self.x = torch.tensor(np.concatenate([real_fcs, fake_fcs]), dtype=torch.float32)
        self.y = torch.tensor(np.concatenate([np.ones(len(real_fcs)), np.zeros(len(fake_fcs))]), dtype=torch.float32).unsqueeze(1)

    def __len__(self):  return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def build_gt_generator(node_size=5, g=80, g_EE=3.5, g_EI=0.42, step=0.05, tp=50):
    # sc = torch.tensor(np.load("exp1_outputs/ring_sc.npy"), dtype=torch.float32, device=DEVICE)
    ring = torch.zeros(node_size, node_size)
    for i in range(node_size):
        ring[i, (i+1) % node_size] = np.random.randn()        # random sign / weight
    sc = ring / torch.norm(ring, p='fro')
    sim = RWWSubjectSimulator(sc, node_size=node_size, TP_per_window=tp,
                              fit_g_EE=False, fit_g_EI=False, use_fic=False,
                              g_EE_init=g_EE, g_EI_init=g_EI,
                              g_init=g, step_size=step)
    sim.model.to(DEVICE)
    return sim


def build_fake_generator(sc_np, g_range, gEE_range, gEI_range, tp=50, step=0.05):
    g     = random.uniform(*g_range)
    g_EE  = random.uniform(*gEE_range)
    g_EI  = random.uniform(*gEI_range)
    print(f"g={g} g_EE={g_EE} g_EI={g_EI}")
    sim   = RWWSubjectSimulator(torch.tensor(sc_np, dtype=torch.float32, device=DEVICE),
                                node_size=sc_np.shape[0], TP_per_window=tp,
                                fit_g_EE=False, fit_g_EI=False, use_fic=False,
                                g_EE_init=g_EE, g_EI_init=g_EI,
                                g_init=g, step_size=step)
    sim.model.to(DEVICE)
    return sim


def train_loop(args):
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    writer  = SummaryWriter(log_dir=out_dir / "tb")
    
    gt_sim = build_gt_generator(step=args.step, tp=args.chunk)
    real_fcs = simulate_fc(gt_sim, chunk_len=args.chunk, n_chunks=args.real_samples)
    np.save(out_dir/"real_fc.npy", real_fcs)
    fig = sns.heatmap(gt_sim.sc, cmap="viridis").get_figure()
    fig.tight_layout()
    plt.savefig(out_dir/"sc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved {out_dir}/sc.png")

    fake_fcs = []
    for _ in range(args.n_fake_subj):
        sc_np = make_ring_sc(noise_std=args.sc_noise)
        fake_sim = build_fake_generator(sc_np,
                                        g_range=args.g_range,
                                        gEE_range=args.gEE_range,
                                        gEI_range=args.gEI_range,
                                        tp=args.chunk, step=args.step)
        fake_mats = simulate_fc(fake_sim, chunk_len=args.chunk, n_chunks=args.fake_samples_per_subj)
        fake_fcs.append(fake_mats)
    fake_fcs = np.concatenate(fake_fcs, axis=0)
    np.save(out_dir/"fake_fc.npy", fake_fcs)

    # Dataset
    ds = FCDataset(real_fcs, fake_fcs)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True)

    disc = FCCNNDisc(n_nodes=gt_sim.model.node_size).to(DEVICE)
    optD = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=(0.5, 0.999))
    bce = torch.nn.BCELoss()

    global_step = 0
    for epoch in range(args.epochs):
        disc.train()
        epoch_loss = 0.0
        for x, y in dl:
            x = x.to(DEVICE); y = y.to(DEVICE)
            optD.zero_grad()
            loss = bce(disc(x), y)
            loss.backward(); optD.step()
            epoch_loss += loss.item()
            writer.add_scalar("train/batch_loss", loss.item(), global_step)
            global_step += 1

        mean_loss = epoch_loss / len(dl)
        writer.add_scalar("train/epoch_loss", mean_loss, epoch)
        print(f"[{time.asctime()}] epoch {epoch+1}/{args.epochs}  loss={mean_loss:.4f}")

    torch.save(disc.state_dict(), out_dir/"disc_toy.pt")
    with open(out_dir/"meta.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"[+] Discriminator saved to {out_dir/'disc_toy.pt'}")


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--chunk",            type=int,   default=50,   help="BOLD window length (TR)")
    p.add_argument("--real_samples",     type=int,   default=200,  help="#FCs from ground-truth")
    p.add_argument("--n_fake_subj",      type=int,   default=20,   help="#synthetic subjects")
    p.add_argument("--fake_samples_per_subj", type=int, default=40)
    p.add_argument("--batch",            type=int,   default=32)
    p.add_argument("--epochs",           type=int,   default=30)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--step",             type=float, default=0.05)
    p.add_argument("--sc_noise",         type=float, default=0.05)
    # parameter ranges for fake sims
    p.add_argument("--g_range",   nargs=2, type=float, default=[10, 150])
    p.add_argument("--gEE_range", nargs=2, type=float, default=[1.0, 5.5])
    p.add_argument("--gEI_range", nargs=2, type=float, default=[0.3, 0.6])
    p.add_argument("--out_dir",   type=str, default="exp2_outputs")
    return p

if __name__ == "__main__":
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    args = get_parser().parse_args()
    train_loop(args)
