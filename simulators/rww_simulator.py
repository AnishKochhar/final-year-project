import torch, numpy as np
from whobpyt.wong_wang import RNNRWW
from whobpyt.ParamsRWW import ParamsRWW
from whobpyt.datatypes import par
from whobpyt.data_loader import DEVICE

class RWWSubjectSimulator:
    """ Wrapper: one subject = one SC = one model instance """
    def __init__(self, sc: torch.Tensor, node_size: int, tr: float = 0.75,
                 TP_per_window: int = 50, step_size: float = 0.05,
                 sampling_size: int = 5, g_init: float = 50, g_init_var: float = 0.1,
                 g_EE_init: float = 3.5, g_EE_init_var: float = 1 / np.sqrt(50),
                 g_EI_init: float = 0.42, g_EI_init_var: float = 1 / np.sqrt(50),
                 g_IE_init: float = 0.42, g_IE_init_var: float = 1 / np.sqrt(50),
                 fit_g: bool = True, fit_g_IE: bool = True, fit_g_EI: bool = True, 
                 fit_g_EE: bool = True, fit_gains: bool = True, use_bifurcation: bool = True,
                 std_in: float = 0.00, std_out: float = 0.00, I_0: float = 0.2,
                 use_fic: bool = True, shared_params = None):
        """
        Supports subject-specific deviations from shared cohort mean mu
            shared_params: Optional Dict[str, torch.nn.Parameter]
                Internal model will  re-use these tensors - enabling population-level training
        """
        self.sc = sc
        self.node_size = node_size  
        self.TP = TP_per_window
        self.use_fic = use_fic

        if self.use_fic:
            beta = self.sc.sum(dim=1)
            alpha = 0.75
            g_fic = (alpha * g_init * beta + 1.0).to(dtype=torch.float32, device=DEVICE).unsqueeze(1)
            kappa_par = par(1.0, fit_par=True)
        else:
            g_fic = torch.zeros((node_size, 1), device=DEVICE)
            kappa_par  = par(1.0, fit_par=False)          # dummy

        if shared_params is not None:
            eps = torch.rand(1, device=DEVICE) # one sample per subject
            g_val = shared_params["g_mean"].value().clone() + eps * shared_params["g_log_sig"].value().clone()
            g_EE_val = shared_params["g_EE_mean"].value().clone() + eps * shared_params["g_EE_log_sig"].value().clone()
            g_EI_val = shared_params["g_EI_mean"].value().clone() + eps * shared_params["g_EI_log_sig"].value().clone()
            
            self.params = ParamsRWW(
                g      = par(g_val,  fit_par=fit_g, device=DEVICE),
                g_EE   = par(g_EE_val,fit_par=fit_g_EE, device=DEVICE),
                g_EI   = par(g_EI_val,fit_par=fit_g_EI, device=DEVICE),
                g_IE   = par(g_IE_init, fit_par=fit_g_IE and (not use_fic), device=DEVICE),
                g_FIC  = par(g_fic, fit_par=use_fic, device=DEVICE),
                kappa  = kappa_par,
                I_0     = par(I_0),
                std_in  = par(std_in),
                std_out = par(std_out)
            )
            # Store raw valyes for PopulationFitter to build KL terms
            self.mu  = shared_params
        else:
            self.params = ParamsRWW(
                g       = par(g_init, g_init, g_init_var, fit_g, fit_g),   # default values 
                g_EE    = par(g_EE_init, g_EE_init, g_EE_init_var, fit_par=fit_g_EE),
                g_EI    = par(g_EI_init, g_EI_init, g_EI_init_var, fit_par=fit_g_EI),
                g_IE    = par(g_IE_init, g_IE_init, g_IE_init_var, fit_par=fit_g_IE and (not use_fic)),         # scalar
                g_FIC   = par(g_fic, fit_par=use_fic),              # vector
                kappa   = kappa_par,
                I_0     = par(I_0),
                std_in  = par(std_in),
                std_out = par(std_out)
            )
            self.mu = None

        self.model = RNNRWW(node_size=node_size, TRs_per_window=TP_per_window, use_Bifurcation=use_bifurcation,
                            step_size=step_size, sampling_size=sampling_size, 
                            tr=tr, sc=sc, use_fit_gains=fit_gains, params=self.params, use_fic=use_fic)
        

    def forward_window(self, x0, hE0):
        ext = torch.zeros(self.node_size, self.model.steps_per_TR, self.TP, device=DEVICE)
        return self.model(ext, x0, hE0)


    def simulate(self, u=0, num_windows=1, base_window_num=0, transient_num=10):
        """ Run a full-length simulation and return (ts_sim, fc_sim) """
        print(f"[Simulator] Simulating {num_windows} windows ")
        X  = self.model.createIC(ver=1).to(DEVICE)
        hE = self.model.createDelayIC(ver=1).to(DEVICE)

        win_data = {n: [] for n in set(self.model.state_names + self.model.output_names)}

        u_hat = np.zeros(
            (self.model.node_size,
             self.model.steps_per_TR,
             (base_window_num + num_windows) * self.model.TRs_per_window))
        u_hat[:, :, base_window_num * self.model.TRs_per_window:] = u

        for win in range(base_window_num + num_windows):
            ext = torch.tensor(
                u_hat[:, :, win * self.model.TRs_per_window:(win + 1) * self.model.TRs_per_window],
                dtype=torch.float32, device=DEVICE)

            next_win, hE_new = self.model(ext, X, hE)

            if win >= base_window_num:
                for k in win_data:
                    win_data[k].append(next_win[k].detach().cpu().numpy())

            X, hE = next_win['current_state'].detach(), hE_new.detach()

        for k in win_data:
            win_data[k] = np.concatenate(win_data[k], axis=1)

        ts_sim = win_data[self.model.output_names[0]]
        fc_sim = np.corrcoef(ts_sim[:, transient_num:])

        return ts_sim, fc_sim
