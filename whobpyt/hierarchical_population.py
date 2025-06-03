""" Joint Fitting with subject-specific deltas and cohort means """

import torch, random, numpy as np
from typing import Sequence
from whobpyt.custom_cost_RWW import CostsRWW
from simulators.rww_simulator import RWWSubjectSimulator
from whobpyt.data_loader import DEVICE

class HierarchicalPopulationFitter:
    def __init__(self,
                 sims: Sequence[RWWSubjectSimulator],          # simulators (independent params)
                 emp_FCs: Sequence[Sequence[torch.Tensor]],    # list[list[FC tensors]]
                 means: dict,                                  # shared parameter means (par instances)
                 *,
                 lr_subject: float = .05, lr_mean: float = .05,
                 lr_hyper: float = .05, lambda_kl: float = 1.0,  # weight of KL/L2 shrinkage
                 sigma: float = 1.0):                            # prior std

        assert len(sims) == len(emp_FCs)
        self.sims = sims
        self.emp_FCs = emp_FCs
        self.means = means
        self.lambda_kl = lambda_kl
        self.sigma2 = sigma ** 2

        self.cost_fns = [CostsRWW(sim.model) for sim in sims]

        # Collect parameters
        subj_params, hyper_params = set(), set()
        for sim in sims:
            subj_params.update(sim.model.params_fitted['modelparameter'])
            hyper_params.update(sim.model.params_fitted['hyperparameter'])
        # Cohort means
        mean_params = [p.val for k, p in means.items() if "mean" in k]
        sig_params = [p.val for k, p in means.items() if "sig" in k]
        subj_params.difference(mean_params)
        subj_params.difference(sig_params)

        print(f"[Population Fitter] Fitting {len(subj_params)} model parameters and {len(hyper_params)} hyperparameters")

        # optimisers
        self.opt_subject = torch.optim.Adam(subj_params, lr=lr_subject, eps=1e-7)
        self.opt_mean    = torch.optim.Adam(mean_params, lr=lr_mean * 2, eps=1e-7)
        self.opt_sig     = torch.optim.Adam(sig_params, lr=lr_mean * 2, eps=1e-7)
        self.opt_hyper   = None
        if len(hyper_params) > 0:
            self.opt_hyper = torch.optim.Adam(hyper_params, lr=lr_hyper, eps=1e-7)

        # mask for FC correlation display
        N = sims[0].model.output_size
        self.mask_e = np.tril_indices(N, -1)

    def log_population_state(self):
        means = {k: p.value().item() for k, p in self.means.items() if "mean" in k}
        sigs  = {k: p.value().item() for k, p in self.means.items() if "sig" in k}
        print(f"μ = {means}  σ = {sigs}")

        g_subj = [sim.model.params.g.value().item() for sim in self.sims]
        print(" subject g mean±sd = %.3f ± %.3f" % (np.mean(g_subj), np.std(g_subj)))


    def _kl_term(self, sim: RWWSubjectSimulator):
        kl = 0.0
        for name in ("g", "g_EE", "g_EI"):
            theta = getattr(sim.model.params, name).val
            mu    = self.means[f"{name}_mean"].value()
            sig   = self.means[f"{name}_log_sig"].value()
            var = torch.exp(2 * sig)
            kl += torch.mean(0.5 * ((theta - mu) ** 2 / var + 2 * sig - 1))
        return self.lambda_kl * kl

    def train_epoch(self, *, windows_per_subj: int = 4, clip_grad: float = 1.0,
                    max_chunks: int | None = None, log_every: bool = True):

        # zero grads
        self.opt_subject.zero_grad(set_to_none=True)
        self.opt_mean.zero_grad(set_to_none=True)
        if self.opt_hyper is not None:
            self.opt_hyper.zero_grad(set_to_none=True)

        # stats holders
        epoch_data_loss, epoch_kl, epoch_items = 0.0, 0.0, 0

        for sim, cost_fn, full_bank in zip(self.sims, self.cost_fns, self.emp_FCs):
            bank = full_bank if max_chunks is None else full_bank[:max_chunks]
            batch = random.sample(bank, k=min(windows_per_subj, len(bank)))

            with torch.no_grad():
                for name in ("g", "g_EE", "g_EI"):
                    mu  = self.means[f"{name}_mean"].value()
                    log_sigma = self.means[f"{name}_log_sig"].value()
                    sigma  = log_sigma.exp()
                    eps  = torch.randn_like(mu)
                    sample = mu + eps * sigma
                    getattr(sim.model.params, name).val.data.copy_(sample)


            for fc_emp in batch:
                X  = sim.model.createIC(ver=0).to(DEVICE)
                hE = sim.model.createDelayIC(ver=0).to(DEVICE)
                ext = torch.zeros(sim.model.node_size, sim.model.steps_per_TR,
                                   sim.model.TRs_per_window, device=DEVICE)
                win_out, _ = sim.model(ext, X, hE)
                data_loss = cost_fn.loss(win_out, fc_emp)
                kl_loss   = self._kl_term(sim)
                loss = data_loss + kl_loss
                loss.backward()
                with torch.no_grad():
                    sim.model.params.g_EI.val.clamp_(0.0, 1.5)

                epoch_data_loss += data_loss.item()
                epoch_kl        += kl_loss.item()
                epoch_items     += 1

        torch.nn.utils.clip_grad_norm_(self.opt_subject.param_groups[0]['params'], clip_grad)
        self.opt_subject.step()
        self.opt_mean.step()
        self.opt_sig.step()
        if self.opt_hyper is not None:
            self.opt_hyper.step()

        if log_every:
            mean_data = epoch_data_loss / max(epoch_items, 1)
            mean_kl   = epoch_kl        / max(epoch_items, 1)
            print(f"data loss = {mean_data:.3f}   KL = {mean_kl:.3f}   total = {mean_data+mean_kl:.3f}")
            print("μ parameters:",
                  "g =",     self.means["g_mean"].value().item(),
                  "g_EE =",  self.means["g_EE_mean"].value().item(),
                  "g_EI =",  self.means["g_EI_mean"].value().item())
            self.log_population_state()

        return (epoch_data_loss + epoch_kl) / max(epoch_items, 1)
