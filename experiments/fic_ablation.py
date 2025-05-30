""" Experiment 4 - vector FIC  vs scalar g_IE """

import argparse, json, numpy as np, torch, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path

from whobpyt.data_loader import BOLDDataLoader, DEVICE
from simulators.rww_simulator import RWWSubjectSimulator
from whobpyt.custom_cost_RWW import CostsRWW
from whobpyt.modelfitting import Model_fitting

