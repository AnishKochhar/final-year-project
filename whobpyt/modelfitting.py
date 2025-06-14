"""
Authors: Zheng Wang, John Griffiths, Andrew Clappison, Hussain Ather, Kevin Kadak
Neural Mass Model fitting module for model fitting using pytorch
"""

import numpy as np  # for numerical operations
import torch
import torch.optim as optim
import pickle, random

from whobpyt.models.fc_cnn_disc import FCCNNDisc, LightFCCNN
from whobpyt.datatypes import Recording
from whobpyt.datatypes.AbstractFitting import AbstractFitting
from whobpyt.datatypes.AbstractNMM import AbstractNMM
from whobpyt.datatypes.AbstractLoss import AbstractLoss
from whobpyt.datatypes.outputs import TrainingStats
from whobpyt.arg_type_check import method_arg_type_check
from whobpyt.utils.plotter import heatmap_fc
from whobpyt.data_loader import DEVICE

    
def d_loss_fn(real_scores, fake_scores):
    return (torch.relu(1. - real_scores).mean() + torch.relu(1. + fake_scores).mean())

def g_loss_fn(fake_scores):
    return -fake_scores.mean()


class Model_fitting(AbstractFitting):
    """
    This Model_fitting class is able to fit resting state data or evoked potential data 
    for which the input training data is empty or some stimulus to one or more NMM nodes,
    and the label is an associated empirical neuroimaging recording.
    
    Studies which consider different kinds of input, such as if SC or some other variable
    is associated with an empirical recording, must use a different fitting class. 
    
    Attributes
    ----------
    model: AbstractNMM 
        Whole Brain Model to Simulate
    cost: AbstractLoss
        A particular objective function which the model will be optimized for. 
    trainingStats: TrainingStats
        Information about objective function loss and parameter values over training windows/epochs
    lastRec: Recording
        The last simulation of fitting(), evaluation(), or simulation()
    device : torch.device
        Whether the fitting is to run on CPU or GPU
    """

    def __init__(self, model: AbstractNMM, cost: AbstractLoss, use_adv=False, lambda_adv=0.1, disc_path=None, device = torch.device('cpu')):
        """
        Parameters
        ----------
        model: AbstractNMM 
            Whole Brain Model to Simulate
        cost: AbstractLoss
            A particular objective function which the model will be optimized for. 
        device : torch.device
            Whether the fitting is to run on CPU or GPU
        """
        method_arg_type_check(self.__init__) # Check that the passed arguments (excluding self) abide by their expected data types
        
        self.model = model
        self.cost = cost
        
        self.device = device
        
        self.trainingStats = TrainingStats(self.model)
        self.lastRec = None #A dictionary or Recordings of the last simulation preformed (either training or evaluation)
        
        # sued for early stopping
        self.best_loss = 5000
        self.min_delta = 0.01
        self.wait = 0
        self.stop = False
        self.patience = 5

        self.use_adv = use_adv
        self.lambda_adv = lambda_adv
        # if self.use_adv:
        self.disc = LightFCCNN(model.output_size).to(DEVICE)
        if disc_path is not None:
            self.disc.load_state_dict(torch.load(disc_path, map_location=DEVICE))
        else:
            print(f"[Model Fitting] No discriminator loaded!")
        self.disc.eval()    

        #self.u = None #This is the ML "Training Input"                
        #self.empTS = ts #This is the ML "Training Labels" - A list
    
    def print_gradients(self, params_dict, label=""):
        print(f"\n--- Gradients Report [{label}] ---")
        for group_name, param_list in params_dict.items():
            print(f"\n[{group_name.upper()}]")
            for i, param in enumerate(param_list):
                if param.grad is None:
                    print(f"  Param {i}: grad is None")
                    continue
                grad = param.grad.detach()
                print(f"  Param {i}: shape={tuple(grad.shape)}, mean={grad.mean().item():.4e}, std={grad.std().item():.4e}, max={grad.max().item():.4e}, min={grad.min().item():.4e}")

    def save(self, filename):
        """
        Parameters
        ----------
        filename: String
            filename to use when saving object to file
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def adv_weight(self, epoch, warm=5, max_lambda=0.5):
        if epoch < warm: return max_lambda * (epoch + 1) / warm
        return max_lambda


    def early_stop(self, curr_loss):
        if curr_loss < self.best_loss - self.min_delta:
            self.best_loss = curr_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stop = True
        return self.stop

    def train(self, u, empFcs: list, 
              num_epochs: int, num_windows: int, learningrate: float = 0.05, 
              lr_2ndLevel: float = 0.05, lr_scheduler: bool = False, 
              early_stopping = False, softplus_threshold = 20, max_chunks = None,
              disc_train_length = 400):
        """
        Parameters
        ----------
        u: type
           This stimulus is the ML "Training Input" 
        empFc: list of Functional Connectivity Matrices
            This is the ML "Training Labels"
        num_epochs: int
            the number of times to go through the entire training data set
        num_windows: int
            Number of windows in empirical time series.             
        learningrate: float
            rate of gradient descent
        lr_2ndLevel: float
            learning rate for priors of model parameters, and possibly others
        lr_scheduler: bool
            Whether to use the learning rate scheduler
        """            
        method_arg_type_check(self.train, exclude = ['u']) # Check that the passed arguments (excluding self) abide by their expected data types
        k_stack = disc_train_length / self.model.TRs_per_window
        offset = 1#np.random.randint(k_stack)
        print(f"offset = {offset}")
        assert k_stack >= 1 and disc_train_length % self.model.TRs_per_window == 0,\
            "disc_train_length must be an integer multiple of TRs_per_window"

        bce = torch.nn.BCELoss()

        # Define two different optimizers for each group
        modelparameter_optimizer = optim.Adam(self.model.params_fitted['modelparameter'], lr=learningrate, eps=1e-7)
        hyperparameter_optimizer = optim.Adam(self.model.params_fitted['hyperparameter'], lr=lr_2ndLevel, eps=1e-7)

        # Define the learning rate schedulers for each group of parameters
        if lr_scheduler:
            total_steps = 0
            for fc_emp in empFcs:
                # total_steps += int(empRec.length/TPperWindow)*num_epochs
                total_steps += num_windows * num_epochs
        
            # total_steps = self.num_windows*num_epochs
            hyperparameter_scheduler = optim.lr_scheduler.OneCycleLR(hyperparameter_optimizer, 
                                                                    #  lr_hyper, 
                                                                     learningrate,
                                                                     total_steps, 
                                                                     anneal_strategy = "cos")
            hlrs = []
            modelparameter_scheduler = optim.lr_scheduler.OneCycleLR(modelparameter_optimizer, 
                                                                     learningrate, 
                                                                     total_steps, 
                                                                     anneal_strategy = "cos")
            mlrs = []
        
        # initial state
        X = self.model.createIC(ver = 0).to(DEVICE)
        # initials of history of E
        hE = self.model.createDelayIC(ver = 0).to(DEVICE)

        # define masks for getting lower triangle matrix indices
        mask = np.tril_indices(self.model.node_size, -1)
        mask_e = np.tril_indices(self.model.output_size, -1)
        
        print(f"[Trainer] Fitting {len(empFcs)} FC matrices with {num_windows} windows of {self.model.TRs_per_window} TRs")

        # LOOP 1/4: Number of Training Epochs
        for i_epoch in range(num_epochs):

            # Pick random subset of empFCs
            if max_chunks is not None:
                epoch_FCs = random.sample(empFcs, k=min(max_chunks, len(empFcs)))
            else:
                epoch_FCs = empFcs

            # TRAINING_STATS: placeholders for the history of trainingStats
            loss_his = []  # loss placeholder to take the average for the epoch at the end of the epoch

            # LOOP 2/4: Number of Recordings in the Training Dataset
            total_items = len(epoch_FCs)
            for i, fc_emp in enumerate(epoch_FCs, 1): 
                X = self.model.createIC(ver = 0).to(DEVICE)
                hE = self.model.createDelayIC(ver = 0).to(DEVICE)

                # TIME SERIES: Create placeholders for the simulated states and outputs of entire time series corresponding to one recording
                windListDict = {} # A Dictionary with a List of windowed time series
                for name in set(self.model.state_names + self.model.output_names):
                    windListDict[name] = []
                
                # initial the external inputs
                external = torch.tensor(
                    np.zeros([self.model.node_size, self.model.steps_per_TR, self.model.TRs_per_window]),
                    dtype=torch.float32, device=DEVICE)
                
                recon_accum = torch.tensor(0.0, device=DEVICE)
                stack_ts = []

                # LOOP 3/4: Number of windowed segments for the recording
                # for win_idx in range(windowedTS.shape[0]):
                for win_idx in range(num_windows):

                    # if the external not empty
                    if not isinstance(u, int):
                        external = torch.tensor(
                            (u[:, :, win_idx * self.model.TRs_per_window:(win_idx + 1) * self.model.TRs_per_window]), 
                            dtype=torch.float32, device=DEVICE)

                    # LOOP 4/4: The loop within the forward model (numerical solver), which is number of time points per windowed segment
                    next_window, hE_new = self.model(external, X, hE)

                    # Calculating loss
                    recon_loss = self.cost.loss(next_window, fc_emp)
                    recon_accum = recon_accum + recon_loss
                    if win_idx >= offset:
                        stack_ts.append(next_window["bold"])

                    if win_idx >= offset and (win_idx + 1) % k_stack == offset:
                        stacked_bold = torch.cat(stack_ts, dim=1)          # (N, 400)
                        stacked_fc   = torch.corrcoef(stacked_bold)
                        p_fake       = self.disc(stacked_fc[None, None])  # shape (1, 1)
                        # print(f"p_fake = {p_fake.item()}")
                        if self.use_adv:
                            lambda_curr  = self.adv_weight(i_epoch, warm=5, max_lambda=self.lambda_adv)
                            adv_loss     = lambda_curr * g_loss_fn(p_fake)
                        else:
                            adv_loss = torch.tensor(0.0, device=DEVICE)
                        total_loss = recon_accum / k_stack + adv_loss

                        hyperparameter_optimizer.zero_grad()
                        modelparameter_optimizer.zero_grad()
                        total_loss.backward()
                        hyperparameter_optimizer.step()
                        modelparameter_optimizer.step()
                        
                        stack_ts.clear()
                        recon_accum = torch.zeros((), device=DEVICE)
                        # recon_accum.zero_()
                        loss_his.append(total_loss.detach().cpu().item())
                        print(f"recon={recon_loss.item():.4f}  "
                            f"adv={adv_loss.item():.4f}  "
                            f"disc(p)={p_fake.item():.3f}")

                     
                    # TIME SERIES: Put the window of simulated forward model.
                    for name in set(self.model.state_names + self.model.output_names):
                        windListDict[name].append(next_window[name].detach().cpu().numpy())

                    if lr_scheduler:
                        #appending (needed to plot learning rate)
                        hlrs.append(hyperparameter_optimizer.param_groups[0]["lr"])
                        mlrs.append(modelparameter_optimizer.param_groups[0]["lr"])
                        
                        # schedular step 
                        hyperparameter_scheduler.step()
                        modelparameter_scheduler.step()

                    # last update current state using next state...
                    # (no direct use X = X_next, since gradient calculation only depends on one batch no history)
                    X = next_window['current_state'].detach().clone() # dtype=torch.float32
                    hE = hE_new.detach().clone() #dtype=torch.float32

                # TIME SERIES: Concatenate all windows together to get one recording
                for name in set(self.model.state_names + self.model.output_names):
                        windListDict[name] = np.concatenate(windListDict[name], axis=1)

                ts_sim = windListDict[self.model.output_names[0]]
                fc_sim = np.corrcoef(ts_sim[:, 10:]) # skip first 10 TRs

                fc_emp_np = fc_emp.detach().cpu().numpy()
                fc_cor = np.corrcoef(fc_sim[mask_e], fc_emp_np[mask_e])[0, 1]


                # if i_epoch == num_epochs - 1:
                print(f"[{i} / {total_items}] FC_cor: {fc_cor:.3f}") # Pseudo as different windows of the time series have slighly different parameter values
                # heatmap_fc(fc_sim, fc_emp, r=fc_cor, tag=f"test_{i_epoch}", outdir="plots/sim_fc", subdir="debug") # Save FC's of final epoch
                      
                # if lr_scheduler:
                #     print('Modelparam_lr: ', modelparameter_scheduler.get_last_lr()[0])
                #     print('Hyperparam_lr: ', hyperparameter_scheduler.get_last_lr()[0])
                    
            # TRAINING_STATS: Put the updated model parameters into the history placeholders at the end of every epoch.
            # Additing Mean Loss for the Epoch
            curr_loss = np.mean(loss_his)
            self.trainingStats.appendLoss(curr_loss)

            print(
                f"epoch {i_epoch:03d}  "
                f"loss={curr_loss:.4f}  "
                f"g={self.model.params.g.val.item():.3f}  "
                f"g_EE={self.model.params.g_EE.val.item():.3f}  "
                f"g_EI={self.model.params.g_EI.val.item():.3f} "
                f"g_IE={self.model.params.g_IE.val.item():.3f} "
                f"kappa={self.model.params.kappa.val.item():.2f}"
            )

            # NMM/Other Parameter info for the Epoch (a list where a number is recorded every window of every record)            
            trackedParam = {}
            exclude_param = ['gains_con', 'lm'] #This stores SC and LF which are saved seperately
            if(self.model.track_params):
                for par_name in self.model.track_params:
                    var = getattr(self.model.params, par_name)
                    if (var.fit_par):
                        trackedParam[par_name] = var.value().detach().cpu().numpy().copy()
                    if (var.fit_hyper):
                        trackedParam[par_name + "_prior_mean"] = var.prior_mean.detach().cpu().numpy().copy()
                        trackedParam[par_name + "_prior_var"] = var.prior_var.detach().cpu().numpy().copy()
            for key, value in self.model.state_dict().items():
                if key not in exclude_param:
                    trackedParam[key] = value.detach().cpu().numpy().ravel().copy()
            self.trainingStats.appendParam(trackedParam)
            # Saving the SC and/or Lead Field State at Every Epoch
            if self.model.use_fit_gains:
                self.trainingStats.appendSC(self.model.sc_fitted.detach().cpu().numpy())
            if self.model.use_fit_lfm:
                self.trainingStats.appendLF(self.model.lm.detach().cpu().numpy())

            if early_stopping & self.early_stop(curr_loss):
                break
        
        # Saving the last recording of training as a Model_fitting attribute
        self.lastRec = {}
        for name in set(self.model.state_names + self.model.output_names):
            self.lastRec[name] = Recording(windListDict[name], step_size = self.model.step_size) #TODO: This won't work if different variables have different step sizes

    def evaluate(self, empFcs: list, fc_sims : list): 
        """
        Function to evaluate the performance of model.

        Parameters
        ----------
        empFcs: list
            List of empirical Functional Connectivity Matrices
        fc_sims: list
            List of simulated Functional Connectivity Matrices
        -----------

        Returns
        --------
        Pearson's Correaltion coefficient of similarity between empirical and 
        simulated FCs
        """

        method_arg_type_check(self.evaluate) # Check that the passed arguments (excluding self) abide by their expected data types

        fc_cor_his = []  # placeholder (FC correlation) to take the average across the recordings
                
        # LOOP 1/3: Number of Empirical BOLD recordings
        for i, fc_emp in enumerate(empFcs):
            fc_sim = fc_sims[i]

            # define mask for getting lower triangle matrix
            # mask = np.tril_indices(self.model.node_size, -1)
            mask_e = np.tril_indices(self.model.output_size, -1)

            fc_cor = np.corrcoef(fc_sim[mask_e], fc_emp[mask_e])[0, 1]
            
            print('Pearson Correlations: ', fc_cor)
            
            fc_cor_his.append(fc_cor)

            return np.mean(fc_cor)

    def simulate(self, u, num_windows: int, base_window_num: int = 0, transient_num: int = 10):
        """
        Parameters
        ----------
        u : int or Tensor
            external or stimulus
        numTP : int
            The number of time points ot simulate
         TPperWindow: int
            Number of Empirical Time Points per window. model.forward does one window at a time.
        base_window_num : int
            length of num_windows for resting
        transient_num : int
            The number of initial time points to exclude from some metrics
        -----------
        """
        method_arg_type_check(self.simulate, exclude = ['u']) # Check that the passed arguments (excluding self) abide by their expected data types

        # initial state
        X = self.model.createIC(ver = 1)
        # initials of history of E
        hE = self.model.createDelayIC(ver = 1).to(DEVICE)
        
        # Create placeholders for the simulated states and outputs of entire time series corresponding to one recording
        windListDict = {} # A Dictionary with a List of windowed time series
        for name in set(self.model.state_names + self.model.output_names):
            windListDict[name] = []

        u_hat = np.zeros(
            (self.model.node_size,self.model.steps_per_TR,
             base_window_num*self.model.TRs_per_window + num_windows*self.model.TRs_per_window))
        u_hat[:, :, base_window_num * self.model.TRs_per_window:] = u

        # LOOP 1/2: The number of windows in a recording
        for win_idx in range(num_windows + base_window_num):

            # Get the input and output noises for the module.
            external = torch.tensor(
                (u_hat[:, :, win_idx * self.model.TRs_per_window:(win_idx + 1) * self.model.TRs_per_window]),
                dtype=torch.float32)

            # LOOP 2/2: The loop within the forward model (numerical solver), which is number of time points per windowed segment
            next_window, hE_new = self.model.forward(external, X, hE)

            # TIME SERIES: Put the window of simulated forward model.
            if win_idx > base_window_num - 1:
                for name in set(self.model.state_names + self.model.output_names):
                    windListDict[name].append(next_window[name].detach().cpu().numpy())

            # last update current state using next state...
            # (no direct use X = X_next, since gradient calculation only depends on one batch no history)
            X = next_window['current_state'].detach().clone() # dtype=torch.float32
            hE = hE_new.detach().clone() #dtype=torch.float32
        
        # TIME SERIES: Concatenate all windows together to get one recording
        for name in set(self.model.state_names + self.model.output_names):
            windListDict[name] = np.concatenate(windListDict[name], axis=1)
        
        ts_sim = windListDict[self.model.output_names[0]]
        fc_sim = np.corrcoef(ts_sim[:, transient_num:])
        
        # Saving the last recording of training as a Model_fitting attribute   -> NOTE: should it do this for a simulation??
        self.lastRec = {}
        for name in set(self.model.state_names + self.model.output_names):
            self.lastRec[name] = Recording(windListDict[name], step_size = self.model.step_size) #TODO: This won't work if different variables have different step sizes
        
        return ts_sim, fc_sim