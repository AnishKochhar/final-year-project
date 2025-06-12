import os, argparse, random, json
from pathlib import Path
import numpy as np, torch, matplotlib.pyplot as plt, seaborn as sns
from tqdm import trange

from whobpyt.data_loader import BOLDDataLoader, DEVICE
from simulators.rww_simulator import RWWSubjectSimulator
from whobpyt.custom_cost_RWW import CostsRWW
from whobpyt.modelfitting import Model_fitting
from whobpyt.models.fc_cnn_disc import FCCNNDisc, LightFCCNN
from whobpyt.utils.fc_tools import bold_to_fc

def corr_lower(a, b):
    """Pearson r between lower-triangle of two FC matrices"""
    tri = np.tril_indices_from(a, k=-1)
    return np.corrcoef(a[tri], b[tri])[0, 1]

def plot_triplet(mats, titles, suptitle, outdir, filename):
    plt.figure(figsize=(9,3))
    for i, (m, t) in enumerate(zip(mats, titles), 1):
        plt.subplot(1, len(mats), i)
        sns.heatmap(m, vmin=-1, vmax=1, cmap='coolwarm',
                    square=True, cbar=False)
        plt.title(t); plt.axis('off')
    plt.suptitle(suptitle); plt.tight_layout(); plt.savefig(f"{outdir}/{filename}")


def build_sim(loader, subj, *, chunk, g=1200,
              fit_g_EE=False, fit_g_EI=True, use_fic=True):
    sc = loader.get_subject_connectome(subj, norm=True)
    sim = RWWSubjectSimulator(
        sc, sc.shape[0], TP_per_window=chunk,
        fit_g_EE=fit_g_EE, fit_g_IE=not use_fic, fit_g_EI=fit_g_EI,
        use_fic=use_fic, step_size=0.05, g_init=g)
    return sim

def load_simulator(simulator_path):
    sim = torch.load(simulator_path, weights_only=False, map_location=DEVICE)
    sim.model.to(DEVICE)
    print(f"Loaded simulator from {simulator_path}")
    # print("g  =", sim.model.params.g.val.item())
    # print("g_EE =", sim.model.params.g_EE.val.item())
    # print("g_EI =", sim.model.params.g_EI.val.item())
    # print("g_IE =", sim.model.params.g_IE.val.item())
    # print("g_FIC (mean) =", sim.model.params.g_FIC.val.mean().item())
    # print("kappa =", sim.model.params.kappa.val.item())
    return sim

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default='/vol/bitbucket/ank121/fyp/HCP Data')
    ap.add_argument('--subj', type=int, default=18)
    ap.add_argument('--chunk', type=int, default=100)
    ap.add_argument('--disc-train', type=int, default=400)
    ap.add_argument('--weights-dir', default='/vol/bitbucket/ank121/whobpyt/whobpyt/trainers/weights/')
    ap.add_argument('--gen-weights', default='gen_sub018_g1200_chunk100.pt')
    ap.add_argument('--disc-weights', default='disc_sub18.pt')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--lr', type=float, default=0.1)
    ap.add_argument('--mix-other-subj', action='store_true',
                    help='augment training set with windows from random subject')
    args = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0); random.seed(0)

    fmri = os.path.join(args.data_root, "BOLD Timeseries HCP.mat")
    scdir = os.path.join(args.data_root, "distance_matrices")
    dist  = os.path.join(args.data_root, "schaefer100_dist.npy")
    loader = BOLDDataLoader(fmri, scdir, dist, args.chunk)
    loader._split_into_chunks()

    simulator_path = os.path.join(args.weights_dir, args.gen_weights)
    discriminator_path = os.path.join(args.weights_dir, f'disc_sub{args.subj:03d}_len{args.disc_train}.pt')
    discriminator_path = os.path.join(args.weights_dir, f'disc_sub{args.subj:03d}_len{args.disc_train}_hinge.pt')
    
    ts_emp = loader.all_bold[args.subj]
    full_emp = bold_to_fc(ts_emp)

    sim0 = build_sim(loader, args.subj, chunk=args.chunk)
    # sim0.model.load_state_dict(torch.load(simulator_path, map_location=DEVICE))
    # sim0 = load_simulator(simulator_path)
    sim0.model.to(DEVICE).eval()
    # duplicate weights so the two branches start identical
    state_dict_clone = {k: v.clone() for k, v in sim0.model.state_dict().items()}

    disc = LightFCCNN(sim0.model.node_size).to(DEVICE)
    disc.load_state_dict(torch.load(discriminator_path, map_location=DEVICE))
    disc.eval()

    sim_base = build_sim(loader, args.subj, chunk=args.chunk); sim_base.model.to(DEVICE)
    # sim_base = load_simulator(simulator_path)
    cost_base = CostsRWW(sim_base.model,
                        use_rate_reg=True,  lambda_rate=0.05,
                        use_spec_reg=True,  lambda_spec=0.05,
                        use_disp_reg=True,  lambda_disp=0.03)

    
    sim_adv = build_sim(loader, args.subj, chunk=args.chunk); sim_base.model.to(DEVICE)
    # sim_adv = load_simulator(simulator_path)
    cost_adv = CostsRWW(sim_adv.model,
                        use_rate_reg=True,  lambda_rate=0.05,
                        use_spec_reg=True,  lambda_spec=0.05,
                        use_disp_reg=True,  lambda_disp=0.03)

    train_window_length = 900
    train_fc = loader.train_fc_windows(args.subj, win_len=train_window_length)
    train_num_windows = 1500 // args.chunk

    if args.mix_other_subj:
        other = (args.subj + 1) % loader.num_subjects
        train_fc += loader.train_fc_windows(other, win_len=train_window_length)
        print(f"[+] Augmented with windows from subject {other}")


    def fit_model(sim, cost, adv, tag):
        fitter = Model_fitting(sim.model, cost, use_adv=adv, lambda_adv=0.1, disc_path=discriminator_path, device=DEVICE)
        fitter.train(u=0, empFcs=train_fc,
                     num_epochs=args.epochs,
                     num_windows=train_num_windows,
                     learningrate=args.lr, early_stopping=True, disc_train_length=args.disc_train)
        return sim

    print("\n=== Fitting WITH adversarial loss ===")
    sim_adv  = fit_model(sim_adv, cost_adv, True, 'adv')
    
    print("\n=== Fitting WITHOUT adversarial loss ===")
    sim_base = fit_model(sim_base, cost_base, False, 'base')

    def simulate_full(sim, label):
        corrs = []
        print(f'Testing {label}...')
        for _ in range(5):
            n_win = ts_emp.shape[1] // args.chunk
            ts_sim, fc_sim = sim.simulate(u=0, num_windows=n_win, base_window_num=10)
            r = corr_lower(fc_sim, full_emp)
            print(f"r = {r:.3f}", end=" ")
            corrs.append(r)
        corrs = np.array(corrs)
        print(f"\n Corr = {corrs.mean():.3f} +- {corrs.std(ddof=1):.3f}")
        return fc_sim, corrs.mean()

    fc_adv,  r_adv  = simulate_full(sim_adv,  'ADV')
    fc_base, r_base = simulate_full(sim_base, 'BASE')

    SAVE_DIR = Path("exp2"); SAVE_DIR.mkdir(parents=True, exist_ok=True)
    plot_triplet(
        [full_emp, fc_base, fc_adv],
        ['Empirical full-TR',
         f'Generator (no-disc)  r={r_base:.3f}',
         f'Generator (disc)     r={r_adv:.3f}'],
        f"Subject {args.subj} - effect of adversarial loss",
        outdir=SAVE_DIR.name, filename="adv_effect.png")
    
    def show_disc_samples():
        # Sample one FC from sim_adv, one from sim_base, and two different empirical FCs
        with torch.no_grad():
            # Simulated (ADV)
            _, fc_adv = sim_adv.simulate(u=0, num_windows=args.disc_train // args.chunk, base_window_num=10)
            # Simulated (BASE)
            _, fc_base = sim_base.simulate(u=0, num_windows=args.disc_train // args.chunk, base_window_num=10)

        # Empirical: pick two non-overlapping windows of disc_train size
        ts_emp = loader.all_bold[args.subj]
        total_T = ts_emp.shape[1]
        win_len = args.disc_train
        if total_T < 2 * win_len:
            raise ValueError("Not enough timepoints for two non-overlapping empirical FCs")
        # Randomly select two non-overlapping windows
        idx1 = np.random.randint(0, total_T - win_len + 1)
        idx2_choices = list(range(0, idx1 - win_len + 1)) + list(range(idx1 + win_len, total_T - win_len + 1))
        if not idx2_choices:
            idx2 = (idx1 + win_len) % (total_T - win_len + 1)
        else:
            idx2 = np.random.choice(idx2_choices)
        emp_fc1 = bold_to_fc(ts_emp[:, idx1:idx1+win_len])
        emp_fc2 = bold_to_fc(ts_emp[:, idx2:idx2+win_len])

        mats = []
        labels = []

        # Simulated ADV
        mats.append(fc_adv)
        p_adv = disc(torch.tensor(fc_adv, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)).item()
        labels.append(f"Sim ADV\np={p_adv:.3f}")

        # Simulated BASE
        mats.append(fc_base)
        p_base = disc(torch.tensor(fc_base, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)).item()
        labels.append(f"Sim BASE\np={p_base:.3f}")

        # Empirical 1
        mats.append(emp_fc1)
        p_emp1 = disc(torch.tensor(emp_fc1, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)).item()
        labels.append(f"Empirical 1\np={p_emp1:.3f}")

        # Empirical 2
        mats.append(emp_fc2)
        p_emp2 = disc(torch.tensor(emp_fc2, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)).item()
        labels.append(f"Empirical 2\np={p_emp2:.3f}")

        plot_triplet(
            mats, labels,
            "Discriminator confidence (Sim ADV, Sim BASE, Empirical 1, Empirical 2)",
            outdir=SAVE_DIR.name, filename="samples.png"
        )
    show_disc_samples()

    
    summary = dict(chunk=args.chunk, disc_train=args.disc_train,
                   r_base=float(r_base), r_adv=float(r_adv),
                   mix_other_subj=args.mix_other_subj)
    print(summary)
    with open(f'{SAVE_DIR}/adv_vs_base.json', 'w') as fp:
        json.dump(summary, fp, indent=2)
    print("Experiment complete")

if __name__ == '__main__':
    main()
