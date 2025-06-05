"""
Authors: Zheng Wang, John Griffiths, Andrew Clappison, Hussain Ather, Kevin Kadak
Neural Mass Model fitting
module for cost calculation
"""

import numpy as np  # for numerical operations
import torch
from whobpyt.datatypes.parameter import par
from whobpyt.datatypes.AbstractLoss import AbstractLoss
from whobpyt.datatypes.AbstractNMM import AbstractNMM
from whobpyt.cost_FC import CostsFixedFC
from whobpyt.arg_type_check import method_arg_type_check
from whobpyt.data_loader import DEVICE
from whobpyt.models.fc_cnn_disc import FCCNNDisc


class CostsRWW(AbstractLoss):
    def __init__(self, model : AbstractNMM, 
                 use_rate_reg=False, lambda_rate=0.1, rate_target=0.15,
                 use_spec_reg=False, lambda_spec=0.05, spec_target = 1.02, 
                 use_disp_reg=False, lambda_disp=0.05,
                 use_adv=False, lambda_adv=0.1, disc_path=None,
                 log_loss=True):
        self.mainLoss = CostsFixedFC("bold", log_loss=log_loss, device=DEVICE)
        self.simKey = "bold"
        self.model = model

        self.use_rate_reg = use_rate_reg        # firing rate target
        self.lambda_rate = lambda_rate
        self.rate_target = rate_target

        self.use_spec_reg = use_spec_reg        # spectral slope
        self.lambda_spec = lambda_spec
        self.spec_target = spec_target

        self.use_disp_reg = use_disp_reg        # fc dispersion penalty (match variance)
        self.lambda_disp = lambda_disp

        self.use_adv = use_adv
        self.lambda_adv = lambda_adv
        if self.use_adv:
            self.disc = FCCNNDisc(model.output_size).to(model.device)
            if disc_path is not None:
                self.disc.load_state_dict(torch.load(disc_path, map_location=model.device))
            else:
                print(f"[Costs] No discriminator loaded!")
            self.disc.eval()    

    def spectral_exponent_torch(self, ts, fs=1/0.72, f_lo=0.02, f_hi=0.10):
        """
            ts : (N, T) BOLD window, float32
            returns scalar beta (positive => 1/f^beta)
        """
        T = ts.shape[1]
        ts = ts - ts.mean(dim=1, keepdim=True)
        # power spectrum
        fft  = torch.fft.rfft(ts, dim=1)
        psd  = (fft.real**2 + fft.imag**2).mean(dim=0)  # (F,)
        freqs = torch.fft.rfftfreq(T, d=1/fs).to(ts.device)  # Hz
        # band mask
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        x = torch.log(freqs[mask] + 1e-8)
        y = torch.log(psd[mask]   + 1e-8)
        # slope beta = -d y / d x
        beta = -torch.dot(x - x.mean(), y - y.mean()) / torch.dot(x - x.mean(), x - x.mean())
        return beta


    def loss(self, simData: dict, empData: float):
        
        method_arg_type_check(self.loss) # Check that the passed arguments (excluding self) abide by their expected data types
        sim = simData
        emp_fc = empData.to(DEVICE)
        
        model = self.model
        state_vals = sim
        
        # define some constants
        lb = 0.001

        w_cost = 10
        # w_cost = 1

        # define the relu function
        m = torch.nn.ReLU()

        exclude_param = []
        if model.use_fit_gains:
            exclude_param.append('gains_con')

        loss_main = self.mainLoss.loss(sim, emp_fc) # FC correlation

        loss_EI = 0

        E_window = state_vals['E']
        I_window = state_vals['I']
        f_window = state_vals['f']
        v_window = state_vals['v']
        x_window = state_vals['x']
        q_window = state_vals['q']
        if model.use_Gaussian_EI and model.use_Bifurcation:
            loss_EI = torch.mean(model.E_v_inv * (E_window - model.E_m) ** 2) \
                      + torch.mean(-torch.log(model.E_v_inv)) + \
                      torch.mean(model.I_v_inv * (I_window - model.I_m) ** 2) \
                      + torch.mean(-torch.log(model.I_v_inv)) + \
                      torch.mean(model.q_v_inv * (q_window - model.q_m) ** 2) \
                      + torch.mean(-torch.log(model.q_v_inv)) + \
                      torch.mean(model.v_v_inv * (v_window - model.v_m) ** 2) \
                      + torch.mean(-torch.log(model.v_v_inv)) \
                      + 5.0 * (m(model.sup_ca) * m(model.g_IE) ** 2
                               - m(model.sup_cb) * m(model.params.g_IE.value())
                               + m(model.sup_cc) - m(model.params.g_EI.value())) ** 2
        if model.use_Gaussian_EI and not model.use_Bifurcation:
            loss_EI = torch.mean(model.E_v_inv * (E_window - model.E_m) ** 2) \
                      + torch.mean(-torch.log(model.E_v_inv)) + \
                      torch.mean(model.I_v_inv * (I_window - model.I_m) ** 2) \
                      + torch.mean(-torch.log(model.I_v_inv)) + \
                      torch.mean(model.q_v_inv * (q_window - model.q_m) ** 2) \
                      + torch.mean(-torch.log(model.q_v_inv)) + \
                      torch.mean(model.v_v_inv * (v_window - model.v_m) ** 2) \
                      + torch.mean(-torch.log(model.v_v_inv))

        if not model.use_Gaussian_EI and model.use_Bifurcation:
            loss_EI = .1 * torch.mean(
                torch.mean(E_window * torch.log(E_window) + (1 - E_window) * torch.log(1 - E_window) \
                           + 0.5 * I_window * torch.log(I_window) + 0.5 * (1 - I_window) * torch.log(
                    1 - I_window), dim=1)) + \
                      + 5.0 * (m(model.sup_ca) * m(model.params.g_IE.value()) ** 2
                               - m(model.sup_cb) * m(model.params.g_IE.value())
                               + m(model.sup_cc) - m(model.params.g_EI.value())) ** 2

        if not model.use_Gaussian_EI and not model.use_Bifurcation:
            loss_EI = .1 * torch.mean(
                torch.mean(E_window * torch.log(E_window) + (1 - E_window) * torch.log(1 - E_window) \
                           + 0.5 * I_window * torch.log(I_window) + 0.5 * (1 - I_window) * torch.log(
                    1 - I_window), dim=1))

        loss_prior = []

        variables_p = [a for a in dir(model.params) if not a.startswith('__') and (type(getattr(model.params, a)) == par)]
        # get penalty on each model parameters due to prior distribution
        for var_name in variables_p:
            # print(var)
            var = getattr(model.params, var_name)
            if model.use_Bifurcation:
                if var.has_prior and var_name not in ['std_in', 'g_EI', 'g_IE'] and \
                        var_name not in exclude_param:
                    loss_prior.append(torch.sum((lb + m(var.prior_var)) * \
                                                (m(var.val) - m(var.prior_mean)) ** 2) \
                                      + torch.sum(-torch.log(lb + m(var.prior_var)))) #TODO: Double check about converting _v_inv to just variance representation
            else:
                if var.has_prior and var_name not in ['std_in'] and \
                        var_name not in exclude_param:
                    loss_prior.append(torch.sum((lb + m(var.prior_var)) * \
                                                (m(var.val) - m(var.prior_mean)) ** 2) \
                                      + torch.sum(-torch.log(lb + m(var.prior_var)))) #TODO: Double check about converting _v_inv to just variance representation
          
        # Firing rate proxy regulariser
        loss_rate = 0.
        if self.use_rate_reg:
            mean_E = torch.mean(E_window)
            loss_rate = (mean_E - self.rate_target).abs()

        # Spectral exponent regulariser
        loss_spec = torch.tensor(0., device=DEVICE)
        bold_win = state_vals[self.simKey]
        if self.use_spec_reg:
            beta_hat = self.spectral_exponent_torch(bold_win)
            loss_spec = (beta_hat - self.spec_target).abs()

        # FC dispersion regulariser
        loss_disp = torch.tensor(0., device=DEVICE)
        if self.use_disp_reg:
            simFC = torch.corrcoef(bold_win)
            tri = torch.tril(torch.ones_like(simFC), -1).bool()
            sim_std = simFC[tri].std()
            emp_std = emp_fc[tri].std()
            loss_disp = (sim_std - emp_std).abs()

        loss_adv = 0.
        if self.use_adv:
            with torch.no_grad():
                p_fake = self.disc(fc_sim.unsqueeze(0)) # (1, 1)
            print(f"p_fake = {p_fake}", end=" ")
            adv_loss = -torch.log(1. - p_fake + 1e-6)
            print(f" adv loss = {adv_loss.squeeze().item()}")
            adv_loss = adv_loss.squeeze()
    
        # total loss
        # print(f"LOSS: [FC] {loss_main:.4f}  [EI] {loss_EI:.3f}  [Rate] {loss_rate:.3f}  [Spec] {loss_spec:.3f}  [Disp] {loss_disp:.3f}")
        loss = (w_cost * loss_main 
                + sum(loss_prior) 
                + 1 * loss_EI 
                + self.lambda_rate * loss_rate
                + self.lambda_spec * loss_spec
                + self.lambda_disp * loss_disp)

        
        return loss
    

