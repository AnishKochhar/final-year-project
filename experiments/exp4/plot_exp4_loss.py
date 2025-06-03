import matplotlib.pyplot as plt
import numpy as np

def parse_losses_from_log_ordered(log_path="logs/exp4.out"):
    with open(log_path, "r") as f:
        lines = f.readlines()

    losses_per_run = []  # List of loss curves in order
    current_run = []

    for line in lines:
        if "epoch" in line and "loss=" in line:
            try:
                loss = float(line.split("loss=")[-1].split()[0])
                current_run.append(loss)
            except:
                continue
        elif "[RESULTS]" in line or "[Trainer] Fitting" in line:
            # This signals the start of a new run, so store old one
            if current_run:
                losses_per_run.append(current_run)
                current_run = []

    # Catch last run
    if current_run:
        losses_per_run.append(current_run)

    # Assign to FIC or Scalar based on even/odd index
    loss_dict = {"FIC": [], "Scalar": []}
    for i, run in enumerate(losses_per_run):
        if i % 2 == 0:
            loss_dict["FIC"].append(run)
        else:
            loss_dict["Scalar"].append(run)

    print(loss_dict)
    return loss_dict



def plot_loss_curves(loss_dict, save_path="exp4_outputs/loss_curves.png"):
    plt.figure(figsize=(3.6, 2.6))
    
    for label, curves in loss_dict.items():
        max_len = max(len(c) for c in curves)
        padded = np.array([c + [c[-1]]*(max_len - len(c)) for c in curves])
        mean = padded.mean(axis=0)
        std = padded.std(axis=0)
        x = np.arange(len(mean))

        plt.plot(x, mean, label=label)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)

    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("Mean loss per condition")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# Run this in your main
losses = parse_losses_from_log_ordered("logs/exp4.out")
plot_loss_curves(losses)