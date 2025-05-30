import torch, numpy as np, argparse
from whobpyt.data_loader import BOLDDataLoader
from whobpyt.custom_cost_RWW import CostsRWW
from whobpyt.modelfitting import Model_fitting
from simulators.rww_simulator import RWWSubjectSimulator
from whobpyt.data_loader import DEVICE

def run_subject(subj_idx, loader, num_epochs=20, lr=0.05):
    sc = loader.get_subject_connectome(subj_idx)
    sim = RWWSubjectSimulator(sc, node_size=sc.shape[0])
    sim.model.to(DEVICE)

    for n, p in sim.model.named_parameters():
        assert p.device.type == "cuda", f"{n} still on {p.device}"


    # empirical FC list (one per chunk) -----------------
    emp_FCs = []
    for bold_chunk, fc_chunk in loader.iter_chunks_per_subject(subj_idx):
        if not isinstance(fc_chunk, torch.Tensor):
            fc_chunk = torch.as_tensor(fc_chunk, dtype=torch.float32)
        emp_FCs.append(fc_chunk)

    # initialise fitting objects ------------------------
    cost     = CostsRWW(sim.model)
    fitter   = Model_fitting(sim.model, cost, device=DEVICE)

    fitter.train(u=0,
                 empFcs=emp_FCs,
                 num_epochs=num_epochs,
                 num_windows=1,          # RNNRWW already iterates TRs_per_window
                 learningrate=lr)

    # simulate a full run for evaluation
    ts_sim, fc_sim = fitter.simulate(u=0, num_windows=len(emp_FCs))
    avg_emp_fc = np.mean(emp_FCs, axis=0)
    fc_corr    = fitter.evaluate(empFcs=[avg_emp_fc], fc_sims=[fc_sim])

    print(f"subject {subj_idx:02d} | FC-corr = {fc_corr:.3f}")
    return fc_corr

if __name__ == "__main__":
    fmri_filename = "/vol/bitbucket/ank121/fyp/HCP Data/BOLD Timeseries HCP.mat"
    sc_dir = "/vol/bitbucket/ank121/fyp/HCP Data/distance_matrices/"
    distance_matrix_path = "/vol/bitbucket/ank121/fyp/HCP Data/schaefer100_dist.npy"

    
    p = argparse.ArgumentParser()
    p.add_argument("--chunk",  type=int, default=50)
    p.add_argument("--epochs", type=int, default=20)
    args = p.parse_args()

    loader = BOLDDataLoader(fmri_filename, sc_dir, distance_matrix_path, chunk_length=args.chunk)
    results = {}

    for s in range(len(loader.all_SC)):
        results[s] = run_subject(s, loader, num_epochs=args.epochs)

    print("Average FC-corr across subjects:", np.mean(list(results.values())))
