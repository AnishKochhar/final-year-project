import numpy as np

def bold_to_fc(bold: np.ndarray) -> np.ndarray:
    """
    bold : shape (nodes, time) -> returns Pearson FC (nodes,nodes)
    """
    return np.corrcoef(bold)
