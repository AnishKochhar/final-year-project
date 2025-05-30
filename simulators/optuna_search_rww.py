import argparse, json, os, random, time, warnings
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np, torch, optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import matplotlib as opt_viz

# -- whobpyt imports ---------------------------------------------------
from whobpyt.data_loader import DEVICE, BOLDDataLoader
from simulators.rww_simulator import RWWSubjectSimulator
from whobpyt.custom_cost_RWW import CostsRWW
from whobpyt.modelfitting import Model_fitting
from whobpyt.utils.plotter import heatmap_fc, scatter_fc, timeseries

@contextmanager
def trial_logger(trial_id: int):
    """ Write stdout of a single trial to its own logfile """
    log_dir = Path("trial_logs"); log_dir.mkdir(exist_ok=True)
    file = open(log_dir / f"trial_{trial_id:04d}.log", "w")
    with redirect_stdout(file):
        yield
    file.close()


def run_single(loader, subject, trial_id, *, g, step_size, chunk_len, epochs, lr, lambda_rate, lambda_spec, lambda_disp):

    # with trial_logger(trial_id):
    print(f"[TRIAL {trial_id}] subject={subject}  "
          f"g={g:.3f}  step={step_size:.3f}  chunk={chunk_len}  "
          f"lr={lr:.4g}  λ_rate={lambda_rate:.3f}  "
          f"λ_spec={lambda_spec:.3f}  λ_disp={lambda_disp:.3f}")

    loader.chunk_length = chunk_len
    loader._split_into_chunks()

    sc = loader.get_subject_connectome(subject, norm=True)
    sim = RWWSubjectSimulator(sc, node_size=sc.shape[0], step_size=step_size,
                            TP_per_window=chunk_len, g_init=g, use_fic=True)
    sim.model.to(DEVICE)

    emp_FCs = [fc for _, fc in loader.iter_chunks_per_subject(subject)]

    cost = CostsRWW(sim.model, use_rate_reg = lambda_rate > 0, lambda_rate  = lambda_rate,
                            use_spec_reg = lambda_spec > 0, lambda_spec  = lambda_spec,
                            use_disp_reg = lambda_disp > 0, lambda_disp  = lambda_disp)


    fit = Model_fitting(sim.model, cost, device=DEVICE)

    fit.train(u=0, empFcs = emp_FCs, num_epochs = epochs,
            num_windows = 1, learningrate=lr, max_chunks = 20, early_stopping=True)


    # Full-length simulation
    emp_ts = loader.all_bold[subject]
    ts_len = emp_ts.shape[1]
    n_windows = ts_len // sim.model.TRs_per_window
    _, fc_sim = fit.simulate(u=0, num_windows=n_windows)

    emp_FC = torch.tensor(np.corrcoef(emp_ts), dtype=torch.float32, device=DEVICE)
    mask = np.tril_indices(emp_FC.shape[0], -1)
    corr = np.corrcoef(fc_sim[mask], emp_FC.cpu().numpy()[mask])[0, 1]

    return corr, fc_sim, emp_FC.cpu().numpy(), fit.trainingStats

# Optuna objective
def build_objective(loader, args):
    def objective(trial: optuna.trial.Trial) -> float:
        g            = trial.suggest_float("g",             10,   1000, log=True)
        step_size    = trial.suggest_float("step_size",     0.01, 0.10)
        chunk_len    = trial.suggest_categorical("chunk",   [30, 50, 100])
        lr           = trial.suggest_float("lr",            1e-4, 0.1,   log=True)
        lambda_rate  = trial.suggest_float("lambda_rate",   0.0,  0.2)
        lambda_spec  = trial.suggest_float("lambda_spec",   0.0,  0.1)
        lambda_disp  = trial.suggest_float("lambda_disp",   0.0,  0.1)


        if args.subjects_per_trial == 1:
            subj_ids = [random.randrange(loader.num_subjects)]
        else:
            subj_ids = random.sample(range(loader.num_subjects), k=args.subjects_per_trial)

        corrs = []
        for index, subject in enumerate(subj_ids, 1):
            print(f"\tSubject = {subject}\n\tg = {g:.2f}\n\tstep_size = {step_size:.3f}\n\tchunk = {chunk_len}"
                  f"\n\tlr = {lr:.3f}\n\tlambda_rate = {lambda_rate:.2f}\n\tlambda_spec = {lambda_spec:.2f}\n\tlambda_disp = {lambda_disp:.2f}")
            corr, fc_sim, fc_emp, tstats = run_single(loader, subject=subject, trial_id=trial.number, g=g, step_size=step_size,
                                              chunk_len=chunk_len, epochs=args.epochs, lr=lr,
                                              lambda_rate=lambda_rate, lambda_spec=lambda_spec, lambda_disp=lambda_disp)
            # trial.set_user_attr("fc_sim",  fc_sim)
            # trial.set_user_attr("fc_emp",  fc_emp)
            # trial.set_user_attr("stats",   tstats)
            if args.plot:
                subdir = "optuna"
                subtitle = f"g = {g:.2f} step = {step_size:.2f} chunk = {chunk_len} lr = {lr:.3f} {lambda_rate:.2f} {lambda_spec:.2f} {lambda_disp:.2f}"
                heatmap_fc(fc_sim, fc_emp, subj=subject, r=corr, subtitle=subtitle, subdir=subdir)
            loss_vals = [float(v) for v in getattr(tstats, "loss", [])]
            trial.set_user_attr("loss_vals", loss_vals)

            corrs.append(corr)
            print(f"{trial.number}: subject={subject}, corr={corr}")
            trial.report(np.mean(corrs), step=index) # Intermediate value for pruner
            if trial.should_prune():
                raise optuna.TrialPruned()
            
        return float(np.mean(corrs))
    
    return objective

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--study-name", required=True, help="Optuna study name / sqlite filename stem")
    parser.add_argument("--trials",      type=int, default=100)
    parser.add_argument("--n-jobs",      type=int, default=1, help="Parallel Optuna workers (processes)")
    parser.add_argument("--seed",        type=int, default=1234)
    parser.add_argument("--plot",        action="store_true")
    
    parser.add_argument("--epochs",      type=int, default=35)
    parser.add_argument("--subjects-per-trial", type=int, default=1)

    parser.add_argument("--data-root",   type=str, required=True, help="Root folder containing BOLD and SC data")
    parser.add_argument("--distance-matrix", type=str, default="/vol/bitbucket/ank121/fyp/HCP Data/schaefer100_dist.npy")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    fmri_path = os.path.join(args.data_root, "BOLD Timeseries HCP.mat")
    sc_dir = os.path.join(args.data_root, "distance_matrices")

    global loader
    loader = BOLDDataLoader(fmri_path, sc_dir, args.distance_matrix, chunk_length=50)

    storage_path = f"sqlite:///{args.study_name}.db"
    study = optuna.create_study(
        study_name = args.study_name,
        direction  = "maximize",
        sampler    = TPESampler(seed=args.seed),
        pruner     = MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        storage    = storage_path,
        load_if_exists=True
    )

    objective = build_objective(loader, args)

    print(f"[INFO] Launching optimisation: {args.trials} trials, {args.n_jobs} workers")
    study.optimize(objective, n_trials=args.trials, n_jobs=args.n_jobs, timeout=None, show_progress_bar=True)

    best = {"best_params": study.best_params,
            "best_value" : study.best_value,
            "n_trials"   : len(study.trials),
            "datetime"   : time.strftime("%Y-%m-%d %H:%M:%S")}
    
    with open(f"optuna_{args.study_name}.json", "w") as fp:
        json.dump(best, fp, indent=2)

    print("\n[RESULT] Best trial:")
    print(json.dumps(best, indent=2))

    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)

    # Optimisation history
    fig1 = opt_viz.plot_optimization_history(study)
    fig1.gca().set_title("Optimisation history - FC correlation ↑")
    fig1.savefig(fig_dir / f"{args.study_name}_history.png", dpi=300, bbox_inches="tight")

    # Parallel coordinates
    fig2 = opt_viz.plot_parallel_coordinate(study, params=[
        "g", "step_size", "chunk", "lr",
        "lambda_rate", "lambda_spec", "lambda_disp"])
    fig2.gca().set_title("Parallel-coordinate view of search space")
    fig2.savefig(fig_dir / f"{args.study_name}_parallel.png", dpi=300, bbox_inches="tight")

    print(f"[INFO] Figures saved to {fig_dir.resolve()}")

    best_trial = study.best_trial
    loss_vals  = best_trial.user_attrs.get("loss_vals", [])
    # fc_sim  = best_trial.user_attrs["fc_sim"]
    # fc_emp  = best_trial.user_attrs["fc_emp"]
    # tstats  = best_trial.user_attrs["stats"]

    # heatmaps & scatter
    # heatmap_fc(fc_sim, fc_emp, subj=None, r=best_trial.value, tag="best_fc", subdir="plots")
    # scatter_fc(fc_sim, torch.tensor(fc_emp), subj=None, r=best_trial.value, tag="best_fc_scatter", subdir="plots")

    # emp_ts_full = loader.all_bold[0][:, :ts_sim.shape[1]]  # safe length
    # timeseries(ts_sim, emp_ts_full, tr=0.75, tag="best_timeseries", subdir="plots")

    # loss curve
    plt.figure()
    plt.plot(loss_vals)
    plt.xlabel("Training window"); plt.ylabel("Loss")
    plt.title("Best-trial loss curve")
    plt.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    plt.savefig("plots/best_loss_curve.png", dpi=300)
    plt.close()
    print("[INFO] Best-trial figures written to ./plots/")



if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        main()
