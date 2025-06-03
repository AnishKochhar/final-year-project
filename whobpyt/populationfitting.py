import torch, random, numpy as np
from typing import Sequence
from whobpyt.custom_cost_RWW import CostsRWW
from whobpyt.data_loader import DEVICE


class PopulationFitter:
    """ 
        Joint optimisation of shared parameter set (g, g_EE, g_EI) across n subjects
    """

    def __init__(self,
                 sims: Sequence,                    # list[RWWSubjectSimulator]
                 emp_FCs: Sequence[Sequence],       # list[list[torch.Tensor]]
                 lr_model: float = .05, lr_hyper: float = .05,
                 lambda_rate=0., lambda_spec=0., lambda_disp=0.):
        assert len(sims) == len(emp_FCs)
        self.sims = sims
        self.emp_FCs = emp_FCs

        self.cost_fns  = []
        for sim in sims:
            self.cost_fns.append(
                CostsRWW(sim.model,
                         use_rate_reg = lambda_rate>0, lambda_rate=lambda_rate,
                         use_spec_reg = lambda_spec>0, lambda_spec=lambda_spec,
                         use_disp_reg = lambda_disp>0, lambda_disp=lambda_disp)
            )

        param_model, param_hyper = set(), set()
        for sim in sims:
            param_model.update(sim.model.params_fitted['modelparameter'])
            param_hyper.update(sim.model.params_fitted['hyperparameter'])

        print(f"[Population Fitter] Fitting {len(param_model)} parameters")
        self.opt_model = torch.optim.Adam(param_model, lr=lr_model, eps=1e-7)

        # create opt_hyper only if needed
        self.opt_hyper = None
        if len(param_hyper) > 0:
            self.opt_hyper = torch.optim.Adam(param_hyper, lr=lr_hyper, eps=1e-7)

        # masks for FC correlation
        self.masks = {}
        N = sims[0].model.output_size
        self.masks['e'] = np.tril_indices(N, -1)

    def train_epoch(self,
                    windows_per_subj: int = 4, clip_grad: float = 1.0, 
                    max_chunks: int | None = None, log_every: bool = True):

        self.opt_model.zero_grad(set_to_none=True)
        if self.opt_hyper is not None:
            self.opt_hyper.zero_grad(set_to_none=True)

        grand_total   = 0.0      # sum of all losses
        grand_items   = 0        # number of windows contributing
        subj_summaries = []      # for pretty printing

        for i, sim in enumerate(self.sims):
            print(f"S{i}", sim.model.params.g.value().item())


        for subj_idx, (sim, cost_fn, full_bank) in enumerate(zip(self.sims, self.cost_fns, self.emp_FCs), 1):

            fc_bank = (full_bank if max_chunks is None else full_bank[:max_chunks]) # Trim if needed

            # Mini-batch sampling
            batch = random.sample(fc_bank, k=min(windows_per_subj, len(fc_bank)))
            subj_loss = 0.0
            
            for fc_emp in batch:
                X  = sim.model.createIC(ver=0).to(DEVICE) # Reset states
                hE = sim.model.createDelayIC(ver=0).to(DEVICE)
                ext = torch.zeros(sim.model.node_size, sim.model.steps_per_TR, sim.model.TRs_per_window, device=DEVICE)

                win_out, hE_new = sim.model(ext, X, hE)

                loss = cost_fn.loss(win_out, fc_emp)
                loss.backward()

                subj_loss += loss.item()
                grand_total += loss.item()
                grand_items += 1

            subj_summaries.append((subj_idx, len(batch), subj_loss))

        torch.nn.utils.clip_grad_norm_(self.opt_model.param_groups[0]['params'], clip_grad)

        self.opt_model.step()
        if self.opt_hyper is not None:
            self.opt_hyper.step()

        if log_every:
            print("─"*60)
            for idx, n_wins, s_loss in subj_summaries:
                print(f"[Subj {idx:02d}]  windows={n_wins:<3d}  "
                      f"sumLoss={s_loss:10.2f}  meanLoss={s_loss/n_wins:8.2f}")
            print("g    =", self.sims[0].model.params.g.value().item(),
                  "g_EE =", self.sims[0].model.params.g_EE.value().item(),
                  "g_EI =", self.sims[0].model.params.g_EI.value().item())
            print("─"*60)

        return grand_total / max(grand_items, 1)

