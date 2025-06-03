import torch
from whobpyt.datatypes import par      
from whobpyt.data_loader import DEVICE

def make_shared_pars(init_g=50., init_gEE=3.5, init_gEI=.42, init_log_sig=-1.):
    """ Define shared par() instances for simulator to trainer population models on """

    def p(val, fit=True):
        return par(val, fit_par=fit, device=DEVICE)
    
    return {
        "g_mean"      : p(init_g),
        "g_log_sig"   : p(init_log_sig),
        "g_EE_mean"   : p(init_gEE),
        "g_EE_log_sig": p(init_log_sig),
        "g_EI_mean"   : p(init_gEI),
        "g_EI_log_sig": p(init_log_sig),
    }
