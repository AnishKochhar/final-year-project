import optuna
from optuna.visualization import matplotlib as opt_viz
import matplotlib.pyplot as plt
from pathlib import Path

# Change this to match your saved study name
study_name = "whobpyt_base_new"
storage_path = f"sqlite:///{study_name}.db"

# Load existing study
study = optuna.load_study(study_name=study_name, storage=storage_path)

# Output directory
fig_dir = Path("plots/optuna")
fig_dir.mkdir(exist_ok=True)

# Optimisation history plot
ax1 = opt_viz.plot_optimization_history(study)
ax1.set_title("Optimisation history - FC correlation ↑")
ax1.figure.savefig(fig_dir / f"{study_name}_history.png", dpi=300, bbox_inches="tight")

# Parallel coordinate plot
ax2 = opt_viz.plot_parallel_coordinate(study, params=[
    "g", "step_size", "chunk", "lr", "lambda_rate", "lambda_spec", "lambda_disp"])
ax2.set_title("Parallel-coordinate view of search space")
ax2.figure.savefig(fig_dir / f"{study_name}_parallel.png", dpi=300, bbox_inches="tight")

print(f"Saved plots to {fig_dir.resolve()}")


best_trial = study.best_trial
loss_vals  = best_trial.user_attrs.get("loss_vals", [])
plt.figure()
plt.plot(loss_vals)
plt.xlabel("Training window"); plt.ylabel("Loss")
plt.title("Best-trial loss curve")
plt.tight_layout()
Path("plots").mkdir(exist_ok=True)
plt.savefig("plots/best_loss_curve.png", dpi=300)
plt.close()
print("[INFO] Best-trial figures written to ./plots/")
