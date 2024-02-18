import argparse
import os
import sys
output_stream = sys.stdout

import cvxpy as cp
import scipy as sc
import numpy as np
import numpy.random as npr
import torch
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
import warnings
warnings.filterwarnings("ignore")
plt.rcParams.update({
    "text.usetex":True,
    
    "font.size":18,
    "font.family": "serif"
})

parser = argparse.ArgumentParser()
parser.add_argument('--foldername', type=str,
                        default="portfolio/", metavar='N')
arguments = parser.parse_args()
foldername = arguments.foldername

nvals = np.array([1000])
# n = 20
m = 8
lower_q = 0.1
upper_q = 0.9
etas = [0.01,0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3]
val_st = {}
val_re = {}
val_st_lower = {}
val_re_lower = {}
val_st_upper = {}
val_re_upper = {}
prob_st = {}
prob_re = {}
val_re_nom_upper = {}
val_re_nom_lower = {}
val_re_nom = {}
prob_re_nom = {}
mu_vals = {}
lam_vals = {}
for N in nvals:
    mu_vals[N] = []
    lam_vals[N] = []
    val_st[N] = []
    val_re[N] = []
    prob_st[N] = []
    prob_re[N] = []
    val_st_lower[N] = []
    val_re_lower[N] = []
    val_st_upper[N] = []
    val_re_upper[N] = []
    val_re_nom[N] = []
    val_re_nom_upper[N] = []
    val_re_nom_lower[N] = []
    prob_re_nom[N] = []
    # for i in range(len(etas)):
    # first = 0
    offset = 8
    for i in [0,1,2,3,4,5,6,7]:
        # dfgrid = pd.read_csv(foldername + f"results{i + offset}/" + f"results/gridmv_{N,m}.csv")
        # dfgrid2= pd.read_csv(foldername + f"results{i+ offset}/" + f"results/gridre_{N,m}.csv")
        dftrain = pd.read_csv(foldername + f"results{i+offset}/" + f"train_{N,m,0}.csv")
        shape = dftrain.shape[0]
        lam_vals[N] = [float(val[8:-4]) for val in dftrain['lam_list']]
        mu_vals[N] = np.array(dftrain['mu'])
        values_st0 = []
        values_re0 = []
        values_st = []
        values_re = []
        values_re2 = []
        tp_prob_st = []
        tp_prob_re = []
        tp_prob_st0 = []
        tp_prob_re0 = []
        tp_prob_re2 = []
        for r in range(15):
            dfgrid = pd.read_csv(foldername + f"results{i+offset}/" + f"gridmv_{N,m,r}.csv")
            dfgrid2 = pd.read_csv(foldername + f"results{i+offset}/" + f"gridre_{N,m,r}.csv")
            ind_s0 = np.absolute(np.mean(np.vstack(dfgrid['Avg_prob_test']),axis = 1)-0.0).argmin()
            ind_r0 = np.absolute(np.mean(np.vstack(dfgrid2['Avg_prob_test']),axis = 1)-0.0).argmin()
            ind_s = np.absolute(np.mean(np.vstack(dfgrid['Avg_prob_test']),axis = 1)-etas[i]).argmin()
            ind_r = np.absolute(np.mean(np.vstack(dfgrid2['Avg_prob_test']),axis = 1)-etas[i]).argmin()
            ind_2 = np.absolute(np.mean(np.vstack(dfgrid2['Eps']),axis = 1)-1).argmin()
            values_st0.append(dfgrid['Test_val'][ind_s0])
            values_re0.append(dfgrid2['Test_val'][ind_r0])
            values_st.append(dfgrid['Test_val'][ind_s])
            values_re.append(dfgrid2['Test_val'][ind_r])
            values_re2.append(dfgrid2['Test_val'][ind_2])
            tp_prob_st.append(dfgrid['Avg_prob_test'][ind_s])
            tp_prob_re.append(dfgrid2['Avg_prob_test'][ind_r])
            tp_prob_st0.append(dfgrid['Avg_prob_test'][ind_s0])
            tp_prob_re0.append(dfgrid2['Avg_prob_test'][ind_r0])
            tp_prob_re2.append(dfgrid2['Avg_prob_test'][ind_2])
        if i==0:
            # first=1
            val_st_lower[N].append(np.quantile(values_st0,lower_q))
            val_st_upper[N].append(np.quantile(values_st0,upper_q))
            val_st[N].append(np.mean(values_st0))
            prob_st[N].append(np.mean(tp_prob_st0))
            
            val_re_lower[N].append(np.quantile(values_re0,lower_q))
            val_re_upper[N].append(np.quantile(values_re0,upper_q))
            val_re[N].append(np.mean(values_re0))
            prob_re[N].append(np.mean(tp_prob_re0))
        
        val_st_lower[N].append(np.quantile(values_st,lower_q))
        val_st_upper[N].append(np.quantile(values_st,upper_q))
        val_st[N].append(np.mean(values_st))
        prob_st[N].append(np.mean(tp_prob_st))
        
        val_re_lower[N].append(np.quantile(values_re,lower_q))
        val_re_upper[N].append(np.quantile(values_re,upper_q))
        val_re[N].append(np.mean(values_re))
        prob_re[N].append(np.mean(tp_prob_re))
        val_re_nom_upper[N].append(np.quantile(values_re2,upper_q))
        val_re_nom_lower[N].append(np.quantile(values_re2,lower_q))
        val_re_nom[N].append(np.mean(values_re2))
        prob_re_nom[N].append(np.mean(tp_prob_re2))

    plt.figure(figsize = (6,3))
    plt.plot(prob_st[N], val_st[N], label = "Mean-Var set", color = "tab:blue")
    plt.plot(prob_re[N], val_re[N], label = "Reshaped set", color = "tab:orange")
    plt.fill_between(prob_st[N],val_st_lower[N],val_st_upper[N], color = "tab:blue", alpha=0.3)
    plt.fill_between(prob_re[N],val_re_lower[N],val_re_upper[N], color = "tab:orange", alpha=0.3)
    plt.plot(prob_re_nom[N],val_re_nom[N],label="Reshaped_orig", color = "tab:green")
    plt.fill_between(prob_re_nom[N],val_re_nom_lower[N],val_re_nom_upper[N], color = "tab:green", alpha=0.3)
    plt.xlabel("Prob. of constraint violation")
    plt.ylabel("Objective value")
    plt.title(f"$m={m}$")
    plt.legend()
    plt.savefig(foldername + f"{m}_{N}_nopar", bbox_inches='tight')
    plt.show()

    plt.figure(figsize = (6,3))
    plt.plot(np.arange(shape), lam_vals[N], label = "lam", color = "tab:blue")
    plt.plot(np.arange(shape), mu_vals[N], label = "mu", color = "tab:orange")
    plt.xlabel("Iters")
    plt.ylabel("value")
    plt.title(f"$m={m}$")
    plt.legend()
    plt.savefig(foldername + f"{m}_{N}_etas_nopar", bbox_inches='tight')
    plt.show()
