import numpy as np, torch, matplotlib.pyplot as plt, seaborn as sns
import os, random

def _np(x): return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

def _save(fig, tag, outdir="plots", subdir=""):
    os.makedirs(f"{outdir}/{subdir}", exist_ok=True)
    path = os.path.join(outdir, subdir, tag + ".png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved {path}")


def heatmap_fc(sim_fc, emp_fc=None, subj=None, r=None, subtitle=None, tag="fc", subdir="", outdir="plots"):
    sim_fc = _np(sim_fc); fig, ax = plt.subplots(1, 2 if emp_fc is not None else 1, figsize=(8, 4))
    sns.heatmap(sim_fc, vmin=-1, vmax=1, cmap="coolwarm", ax=ax if emp_fc is None else ax[0], square=True, cbar_kws={"label": "corr"})
    (ax if emp_fc is None else ax[0]).set_title("Sim FC")
    if emp_fc is not None:
        sns.heatmap(_np(emp_fc), vmin=-1, vmax=1, cmap="coolwarm", ax=ax[1], square=True, cbar_kws={"label": "corr"})
        ax[1].set_title("Emp FC")
    
    supt = []
    if subj is not None: supt.append(f"subj {subj}")
    if r is not None:    supt.append(f"r = {r:.3f}")
    if subtitle is not None: supt.append(subtitle)
    if supt: fig.suptitle(" | ".join(supt))
    _save(fig, f"{tag}_{subj or ''}", subdir=subdir, outdir=outdir)

def scatter_fc(sim_fc, emp_fc, subj=None, r=None, tag="scatter", subdir=""):
    sim_fc = _np(sim_fc); emp_fc = _np(emp_fc)
    mask = np.tril_indices_from(sim_fc, k=-1)

    fig, ax = plt.subplots(figsize=(3,3))
    ax.scatter(emp_fc[mask], sim_fc[mask], s=8, alpha=0.6)
    ax.set_xlabel("Emp FC"); ax.set_ylabel("Sim FC")
    lim = (-1,1); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.plot(lim, lim, ls="--", lw=0.5)
    if r is not None:
        ax.set_title(f"r = {r:.3f}")
    _save(fig, f"{tag}_{subj or ''}", subdir=subdir)

def timeseries(sim_ts, emp_ts, max_nodes=6, tr=0.72, sep=True, tag="ts", subdir="", outdir="plots"):
    """ sep toggles between all channels on same axis and vertically separating them """
    sim_ts, emp_ts = _np(sim_ts), _np(emp_ts)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
    T = sim_ts.shape[1]
    emp_ts_cropped = emp_ts[:, :T] # (N, T)
    panel_data = [("Empirical BOLD Series", emp_ts_cropped), ("Simulated BOLD Series", sim_ts)]
    
    t = np.arange(T) * tr             
    t_axes = [t, t]
    if sep:
        pooled = np.vstack([d[:max_nodes, :] for _, d in panel_data])
        offset = 2.5 * np.nanstd(pooled)
    else:
        offset = 0.0
    
    N = random.sample(range(sim_ts.shape[0]), k=max_nodes)
    for (title, data), ax, t in zip(panel_data, axes, t_axes):
        for i, n in enumerate(N):
            ax.plot(t, data[n] + offset * i, label=f"Node {n}")
        ax.set_title(title)
        ax.set_yticks([])
        ax.legend(ncol=3, fontsize=7, loc="upper right")
    
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    _save(fig, tag, subdir=subdir, outdir=outdir)


def laplacian(L, subj, title=None, subdir=""):
    fig = sns.heatmap(_np(L), cmap="viridis").get_figure()
    fig.tight_layout(); 
    _save(fig, f"{title}_{subj}" if title else f"lap_{subj}", subdir=subdir)


def heatmap_sc(orig_sc, learned_sc, subj=None, tag="sc", subdir=""):
    """ Heatmaps of sc before vs. after training """
    orig_sc, learned_sc = _np(orig_sc), _np(learned_sc)

    # Choose correct colour scales
    vabs = max(abs(orig_sc).max(), abs(learned_sc).max())
    vmin = -vabs if (orig_sc.min() < 0 or learned_sc.min() < 0) else 0.0
    cmap = "coolwarm" if vmin < 0 else "viridis"

    fig, ax = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    sns.heatmap(orig_sc, vmin=vmin, vmax=vabs, cmap=cmap, square=True, ax=ax[0], cbar_kws={"label": "weight"})
    sns.heatmap(learned_sc, vmin=vmin, vmax=vabs, cmap=cmap, square=True, ax=ax[1], cbar_kws={"label": "weight"})

    ax[0].set_title("Original SC"); ax[1].set_title("Learned SC")
    if subj is not None: fig.suptitle(f"Subject {subj}")
    fig.tight_layout()
    _save(fig, tag, subdir=subdir)
