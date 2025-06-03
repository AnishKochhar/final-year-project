import torch, numpy as np, argparse
from whobpyt.data_loader import BOLDDataLoader
from whobpyt.custom_cost_RWW import CostsRWW
from whobpyt.modelfitting import Model_fitting
from simulators.rww_simulator import RWWSubjectSimulator
from whobpyt.data_loader import DEVICE

def run_subject(subj_idx, loader, num_epochs=20,  lr=0.05, g=50, step=0.01, \
                            chunk=30, lam_rate=0.1, lam_spec=0.05, lam_disp=0.05):
    sc = loader.get_subject_connectome(subj_idx, norm=True)
    sim = RWWSubjectSimulator(sc, node_size=sc.shape[0], step_size=step,
                        TP_per_window=chunk, g_init=g, use_fic=True)
    sim.model.to(DEVICE)

    for n, p in sim.model.named_parameters():
        assert p.device.type == "cuda", f"{n} still on {p.device}"

    emp_FCs = [fc for _, fc in loader.iter_chunks_per_subject(subj_idx)]
    
    cost = CostsRWW(sim.model, use_rate_reg = lam_rate > 0, lambda_rate  = lam_rate,
                            use_spec_reg = lam_spec > 0, lambda_spec  = lam_spec,
                            use_disp_reg = lam_disp > 0, lambda_disp  = lam_disp)

    fitter = Model_fitting(sim.model, cost, device=DEVICE)

    fitter.train(u=0, empFcs = emp_FCs, num_epochs = num_epochs,
            num_windows = 1, learningrate=lr, max_chunks = 20, early_stopping=True)

    # Full-length simulation
    emp_ts = loader.all_bold[subj_idx]
    ts_len = emp_ts.shape[1]
    n_windows = ts_len // sim.model.TRs_per_window
    _, fc_sim = fitter.simulate(u=0, num_windows=n_windows)

    emp_FC = torch.tensor(np.corrcoef(emp_ts), dtype=torch.float32, device=DEVICE)
    mask = np.tril_indices(emp_FC.shape[0], -1)
    corr = np.corrcoef(fc_sim[mask], emp_FC.cpu().numpy()[mask])[0, 1]

    print(f"Subject {subj_idx:03d} | FC-corr = {corr:.3f}")
    return corr

if __name__ == "__main__":
    fmri_filename = "/vol/bitbucket/ank121/fyp/HCP Data/BOLD Timeseries HCP.mat"
    sc_dir = "/vol/bitbucket/ank121/fyp/HCP Data/distance_matrices/"
    distance_matrix_path = "/vol/bitbucket/ank121/fyp/HCP Data/schaefer100_dist.npy"

    p = argparse.ArgumentParser()
    p.add_argument("--chunk",  type=int, default=30)
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--g",      type=float, default=36.5)
    p.add_argument("--step",   type=float, default=0.064)
    p.add_argument("--lr",     type=float, default=0.094)
    p.add_argument("--lambda_rate", type=float, default=0.095)
    p.add_argument("--lambda_spec", type=float, default=0.076)
    p.add_argument("--lambda_disp", type=float, default=0.030)
    args = p.parse_args()

    loader = BOLDDataLoader(fmri_filename, sc_dir, distance_matrix_path, chunk_length=args.chunk)
    loader._split_into_chunks()
    
    results = {}
    for s in range(len(loader.all_SC)):
        corr = run_subject(s, loader, num_epochs=args.epochs, lr=args.lr, g=args.g, step=args.step, \
                            lam_rate=args.lambda_rate, lam_spec=args.lambda_spec, lam_disp=args.lambda_disp)
        results[s] = corr

    print("Average FC-corr across subjects:", np.mean(list(results.values())))
