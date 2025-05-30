import torch, numpy as np
from whobpyt.wong_wang import RNNRWW
from whobpyt.ParamsRWW import ParamsRWW
from whobpyt.datatypes import par
from whobpyt.data_loader import DEVICE

class RWWSubjectSimulator:
    """ Wrapper: one subject = one SC = one model instance """
    def __init__(self, sc: torch.Tensor, node_size: int, tr: float = 0.75,
                 TP_per_window: int = 50, step_size: float = 0.05,
                 sampling_size: int = 5, g_init: float = 50,
                 use_fic: bool = True):
        self.sc = sc
        self.node_size = node_size
        self.TP = TP_per_window
        self.use_fic = use_fic

        if self.use_fic:
            beta = self.sc.sum(dim=1)
            alpha = 0.75
            g_fic = (alpha * g_init * beta + 1.0).to(dtype=torch.float32, device=DEVICE).unsqueeze(1)
        else:
            g_fic = torch.full((node_size, 1), 0.42, device=DEVICE)


        self.params = ParamsRWW(
            g       = par(g_init, g_init, 1/10, True, True),   # default values – tune later
            g_EE    = par(3.5, fit_par=True),
            g_EI    = par(0.42, fit_par=True),
            g_IE    = par(0.42, fit_par=False),         # scalar
            g_FIC   = par(g_fic, fit_par=True),         # vector
            I_0     = par(0.2),
            std_in  = par(0.0),
            std_out = par(0.01)
        )

        self.model = RNNRWW(node_size=node_size, TRs_per_window=TP_per_window,
                            step_size=step_size, sampling_size=sampling_size, 
                            tr=tr, sc=sc, use_fit_gains=True, params=self.params, use_fic=use_fic)
        

    def forward_window(self, x0, hE0):
        ext = torch.zeros(self.node_size, self.model.steps_per_TR, self.TP, device=DEVICE)
        return self.model(ext, x0, hE0)
