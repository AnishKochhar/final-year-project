import os, json, argparse, random, numpy as np, torch
import seaborn as sns
from pathlib import Path
from tqdm import trange

from whobpyt.data_loader import BOLDDataLoader, DEVICE
from simulators.rww_simulator import RWWSubjectSimulator
from whobpyt.custom_cost_RWW import CostsRWW
from whobpyt.modelfitting import Model_fitting
from whobpyt.models.fc_cnn_disc import FCCNNDisc
from whobpyt.utils.plotter import heatmap_fc

def build_generator(loader, subj, *, chunk, g0, step, lr, epochs,
                    fit_g_EE=True, fit_g_EI=True, save_path=None):
    sc = loader.get_subject_connectome(subj, norm=True)
    sim = RWWSubjectSimulator(
        sc, sc.shape[0],
        TP_per_window=chunk,
        fit_g_EE=fit_g_EE, fit_g_IE=False, fit_g_EI=fit_g_EI,
        use_fic=False, step_size=step, g_init=g0)
    sim.model.to(DEVICE)

    full_emp = torch.tensor(np.corrcoef(loader.all_bold[subj]), dtype=torch.float32, device=DEVICE)
    cost = CostsRWW(sim.model, use_rate_reg=True, lambda_rate=0.05,
                    use_spec_reg=True, lambda_spec=0.05,
                    use_disp_reg=True, lambda_disp=0.03)
    fitter = Model_fitting(sim.model, cost, device=DEVICE)
    fitter.train(u=0,
                 empFcs=[full_emp],
                 num_epochs=epochs,
                 num_windows=loader.all_bold[subj].shape[1] // chunk,
                 learningrate=lr,
                 early_stopping=True)
    
    if save_path:
        torch.save(sim.model.state_dict(), save_path)
    return sim


def get_empirical_batch(loader, subj, chunk, B):
    """ Return B empirical FCs """
    ts = loader.all_bold[subj]
    idxs = np.random.randint(0, ts.shape[1] - chunk + 1, size=B)
    fc_list = [np.corrcoef(ts[:, i:i+chunk]) for i in idxs]
    return torch.tensor(np.stack(fc_list), dtype=torch.float32, device=DEVICE)


def get_simulated_batch(sim, chunk, n_rep, B):
    """ Run n_rep full simulations, cut into chunks, pick B at random """
    fc_pool = []
    usable = (1200 // chunk) * chunk
    for _ in range(n_rep):
        ts, _ = sim.simulate(u=0, num_windows=usable // chunk, base_window_num=10)
        idxs = np.random.randint(0, ts.shape[1] - chunk + 1, size=max(B // n_rep + 1, 1))
        fc_pool.extend([np.corrcoef(ts[:, j:j+chunk]) for j in idxs])
    fc_pool = np.stack(fc_pool[:B])
    return torch.tensor(fc_pool, dtype=torch.float32, device=DEVICE)

def visualise_disc_output(disc, real_fc, fake_fc, subj=18):
    test_real = real_fc[0].cpu().numpy()
    test_fake = fake_fc[0].cpu().numpy()
    p_fake = float(torch.sigmoid(disc(fake_fc[[0]])).item())

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(test_real, vmin=-1, vmax=1, cmap="coolwarm")
    axs[0].set_title("Real FC")
    axs[0].axis("off")
    axs[1].imshow(test_fake, vmin=-1, vmax=1, cmap="coolwarm")
    axs[1].set_title(f"Fake FC\nDisc output: {p_fake:.3f}")
    axs[1].axis("off")
    plt.suptitle(f"Disc Sanity Check | Subject {subj}")
    plt.tight_layout()
    plt.savefig("plots/disc_train/test.png")

    # Normalised
    real_z = (test_real - test_real.mean()) / test_real.std()
    fake_z = (test_fake - test_fake.mean()) / test_fake.std()
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(real_z, vmin=-1, vmax=1, cmap="coolwarm")
    axs[0].set_title("Real FC (z-scored)")
    axs[0].axis("off")
    axs[1].imshow(fake_z, vmin=-1, vmax=1, cmap="coolwarm")
    axs[1].set_title("Fake FC (z-scored)")
    axs[1].axis("off")
    plt.suptitle("Z-scored FC comparison")
    plt.tight_layout()
    plt.savefig("plots/disc_train/test_norm.png")

def show_batch_fc(real_fc, fake_fc, n=8):
    """ Visualise n real and n fake FC matrices """
    fig, axs = plt.subplots(2, n, figsize=(2*n, 4))
    for i in range(n):
        sns.heatmap(real_fc[i].squeeze(), ax=axs[0, i], vmin=-1, vmax=1, cmap="coolwarm", cbar=False)
        axs[0, i].set_title("Real")
        sns.heatmap(fake_fc[i].squeeze(), ax=axs[1, i], vmin=-1, vmax=1, cmap="coolwarm", cbar=False)
        axs[1, i].set_title("Fake")
    plt.tight_layout()
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/vol/bitbucket/ank121/fyp/HCP Data")
    ap.add_argument("--gen-path", default="discriminator_weights/gen_sub18", help="Load generator weights")
    ap.add_argument("--save-gen", action="store_true")
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--epochs-gen", type=int, default=20)
    ap.add_argument("--epochs-disc", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    # load data
    subj = 18
    chunk = args.chunk
    fmri = os.path.join(args.data_root, "BOLD Timeseries HCP.mat")
    loader = BOLDDataLoader(fmri, os.path.join(args.data_root, "distance_matrices"), os.path.join(args.data_root, "schaefer100_dist.npy"), chunk_length=chunk)
    loader._split_into_chunks()

    generator_save_path = f"{args.gen_path}_{chunk}.pt"
    # Build / load generator
    if args.save_gen and Path(generator_save_path).exists():
        sc = loader.get_subject_connectome(subj, norm=True)
        sim = RWWSubjectSimulator(sc, sc.shape[0], TP_per_window=chunk,
                                  fit_g_EE=True, fit_g_IE=False, fit_g_EI=True,
                                  use_fic=True, step_size=0.05, g_init=1000)
        sim.model.load_state_dict(torch.load(args.gen_path, map_location=DEVICE))
        sim.model.to(DEVICE)
        print(f"[+] Generator loaded from {args.gen_path}")
    else:
        print("[+] Training generator")
        sim = build_generator(loader, subj,
                              chunk=chunk, g0=1000, step=0.05,
                              lr=0.1, epochs=args.epochs_gen,
                              save_path=generator_save_path)
        print(f"[+] Generator saved to {args.gen_path}")

    # build discriminator
    disc = FCCNNDisc(n_nodes=sim.model.node_size).to(DEVICE)
    optD = torch.optim.Adam(disc.parameters(), lr=1e-4, betas=(0.5, 0.999))
    bce = torch.nn.BCELoss()

    # Training loop
    for ep in range(args.epochs_disc):
        pbar = trange(20, desc=f"Disc-epoch {ep+1}/{args.epochs_disc}", leave=False)
        for _ in pbar:
            real_fc = get_empirical_batch(loader, subj, chunk, args.batch)
            fake_fc = get_simulated_batch(sim, chunk, n_rep=5, B=args.batch)
            show_batch_fc(real_fc.cpu().numpy(), fake_fc.cpu().numpy())

            y_real = torch.ones(args.batch, 1, device=DEVICE)
            y_fake = torch.zeros(args.batch, 1, device=DEVICE)

            disc.train(); optD.zero_grad()
            loss_real = bce(disc(real_fc), y_real); loss_fake = bce(disc(fake_fc), y_fake)
            print(f"[Loss] Real = {loss_real.item()}  Fake = {loss_fake.item()}")
            loss = loss_real + loss_fake
            loss.backward(); optD.step()
            pbar.set_postfix(d_loss=float(loss))
            heatmap_fc(fake_fc[0].detach().cpu().numpy(), real_fc[0].detach().cpu().numpy(), outdir="plots", subdir="disc_train")

    torch.save(disc.state_dict(), "discriminator_weights/disc_sub18.pt")
    print("[+] discriminator saved → discriminator_weights/disc_sub18.pt")

    visualise_disc_output(disc, real_fc, fake_fc, subj=subj)


if __name__ == "__main__":
    torch.manual_seed(0); np.random.seed(0)
    main()