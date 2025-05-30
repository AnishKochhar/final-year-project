## Whole-Brain Modelling with PyTorch

This repo uses the [Whole Brain Modelling in PyTorch](https://griffithslab.github.io/whobpyt/about_whobpyt/overview.html) (`whobpyt`) Python package released by the GriffithsLab


### How to run

You can choose to run in the terminal with a `python` command or queue to Slurm with  `sbatch`. The Slurm scripts are stored inside the `bash` directory, with regular Python scripts stored inside `experiments` and `simulators`

#### Experiments

`regularizer_ablation.py`: Toggles 3 additional regularizers, averages over 3 runs, and plots / logs FC correlation for each permutation. The regularizers are: 1) Firing Rate target, 2) Spectral Slope target, and 3) FC Variance penalty. These serve to enforce stronger priors and keep the dynamics in the desired range - at the cost of run time. We show emprically that including our optimizers leads to ~20% better performance

`fic_ablation.py`: Study in effect of scalar g_IE (original `whobpyt` implementation) vs. vectorised FIC proposed by Deco et. al in '_How Local Excitation–Inhibition Ratio Impacts the Whole Brain Dynamics_' (2014) and vectorised by Herzog et. al in '_Neural mass modeling for the masses: Democratizing access to whole-brain biophysical modeling with FastDMF_' (2024). 




### Citations

```
Griffiths JD, Wang Z, Ather SH, Momi D, Rich S, Diaconescu A, McIntosh AR, Shen K. Deep Learning-Based Parameter Estimation for Neurophysiological Models of Neuroimaging Data. bioRxiv. 2022 May 19:2022-05.

Momi D, Wang Z, Griffiths JD. TMS-evoked responses are driven by recurrent large-scale network dynamics. Elife. 2023;12.
```



**Anish Kochhar, June 2025**