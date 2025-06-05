import numpy as np
from pathlib import Path

def save_synthetic_batch(fc_batch, out_dir: Path):
    """ Dump a batch of FC matrices (or BOLD) to *.npy for offline usage """
    out_dir.mkdir(exist_ok=True, parents=True)
    for fc in fc_batch.cpu().numpy():
        fname = out_dir / f"fc_{np.random.randint(1e9)}.npy"
        np.save(fname, fc)
