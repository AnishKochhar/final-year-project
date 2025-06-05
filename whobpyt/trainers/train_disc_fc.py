
import argparse, random, time, json, os
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from tqdm import trange, tqdm

from whobpyt.data_loader import BOLDDataLoader, DEVICE
from whobpyt.models.fc_cnn_disc import FCCNNDisc
from simulators.rww_simulator import RWWSubjectSimulator
from whobpyt.utils.save_synth import save_synthetic_batch

def get_batch(loader, synthetic_files, sims, batch_B):
    """
    Returns tuple (fc_real, fc_fake) shape (B, N, N) float32
    """
    fc_real, fc_fake = [], []
    #  empirical
    subj_ids = random.choices(range(loader.num_subjects), k=batch_B)
    for sid in subj_ids:
        _, fc = random.choice(loader.all_bold[sid])  # pre-split windows
        fc_real.append(fc.cpu().numpy())
    fc_real = torch.tensor(np.stack(fc_real), dtype=torch.float32, device=DEVICE)

    # simulated  
    if synthetic_files is not None:
        chosen = random.sample(synthetic_files, k=batch_B)
        fc_fake = torch.tensor(np.stack([np.load(f) for f in chosen]), dtype=torch.float32, device=DEVICE)
    else: # Generate shorted TS on fly
        for sim in random.choices(sims, k=batch_B):
            _, fc_sim = sim.simulate(u=0, num_windows=10)
            fc_fake.append(fc_sim)
        fc_fake = torch.stack(fc_fake, dim=0)

    return fc_real, fc_fake


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/vol/bitbucket/ank121/fyp/HCP Data")
    p.add_argument("--synthetic-dir", default="/vol/bitbucket/ank121/synthetic_fc")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--window-len", type=int, default=50)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--g-init", type=float, default=50)
    p.add_argument("--dump", action="store_true", help="Saves 'good' (>0.5 from disc) synthetic time series to disk for later re-training")
    args = p.parse_args()

    fmri = os.path.join(args.data_root, "BOLD Timeseries HCP.mat")
    scdir = os.path.join(args.data_root, "distance_matrices")
    dist  = os.path.join(args.data_root, "schaefer100_dist.npy")
    loader = BOLDDataLoader(fmri, scdir, dist, chunk_length=args.window_len)
    loader._split_into_chunks()             


    sims = []
    for sid in range(loader.num_subjects):
        sc = loader.get_subject_connectome(sid, norm=True)
        sims.append(RWWSubjectSimulator(sc, sc.shape[0], TP_per_window=args.window_len,
                                        step_size=0.05, g_init=args.g_init, use_fic=True))
    for sim in sims: sim.model.to(DEVICE)

    if args.synthetic_dir is not None:
        synth_files = list(Path(args.synthetic_dir).glob("*.npy"))
        assert synth_files, "No .npy files found in synthetic-dir"


    disc = FCCNNDisc(n_nodes=sims[0].model.node_size).to(DEVICE)
    opt_D = optim.Adam(disc.parameters(), lr=1e-4, betas=(0.5, 0.999))

    gen_params = {p for sim in sims for p in sim.model.params_fitted['modelparameter']}
    opt_G = optim.Adam(list(gen_params), lr=1e-5) 

    bce_loss = nn.BCELoss()

    log = {"iter": [], "d_loss": [], "g_loss": []}
    out_root = Path("disc_training_logs"); out_root.mkdir(exist_ok=True)

    global_step = 0
    for epoch in range(args.epochs):
        pbar = trange(200, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for _ in pbar:
            real_fc, fake_fc = get_batch(loader, synth_files, sims, args.batch)
            
            disc.train();  opt_D.zero_grad()
            y_real = torch.ones(args.batch, 1, device=DEVICE)
            y_fake = torch.zeros(args.batch, 1, device=DEVICE)

            p_real = disc(real_fc)
            p_fake = disc(fake_fc.detach())
            d_loss = bce_loss(p_real, y_real) + bce_loss(p_fake, y_fake)
            d_loss.backward(); opt_D.step()

            if global_step % 5 == 0:
                opt_G.zero_grad()
                p_fake = disc(fake_fc)               # re-compute
                g_loss = bce_loss(p_fake, y_real)         # non-saturating
                g_loss.backward(); opt_G.step()

            pbar.set_postfix(d_loss=float(d_loss), g_loss=float(g_loss))
            log["iter"].append(global_step)
            log["d_loss"].append(float(d_loss))
            log["g_loss"].append(float(g_loss))
            global_step += 1


            if args.dump and float(p_fake.mean()) > 0.5:
                save_synthetic_batch(fake_fc, out_root / "fc_fake")

    torch.save(disc.state_dict(), out_root / "fc_cnn_disc.pt")
    with open(out_root / "train_log.json", "w") as fp:
        json.dump(log, fp, indent=2)
    
    print(f"Discriminator saved to {out_root}")

if __name__ == "__main__":
    torch.manual_seed(0); np.random.seed(0)
    main()
