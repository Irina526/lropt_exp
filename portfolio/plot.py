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


parser = argparse.ArgumentParser()
parser.add_argument('--foldername', type=str,
                        default="portfolio/", metavar='N')
arguments = parser.parse_args()
foldername = arguments.foldername


nvals = np.array([50,80,100,500,1000,1500,2000,3000,4000,5000])
n = 5
etas = [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3]
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
for N in nvals:
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
    for i in range(len(etas)):
        dfgrid = pd.read_csv(foldername + f"results{i}/" + f"results/gridmv_{N,n}.csv")
        dfgrid2= pd.read_csv(foldername + f"results{i}/" + f"results/gridre_{N,n}.csv")
        if i==0:
            ind_s = np.absolute(np.mean(np.vstack(dfgrid['Avg_prob_test']),axis = 1)-0.0).argmin()
            val_st_lower[N].append(dfgrid['Lower_test'][ind_s])
            val_st_upper[N].append(dfgrid['Upper_test'][ind_s])
            val_st[N].append(dfgrid['Test_val'][ind_s])
            prob_st[N].append(dfgrid['Avg_prob_test'][ind_s])
            ind_r = np.absolute(np.mean(np.vstack(dfgrid2['Avg_prob_test']),axis = 1)-0.0).argmin()
            val_re_lower[N].append(dfgrid2['Lower_test'][ind_r])
            val_re_upper[N].append(dfgrid2['Upper_test'][ind_r])
            val_re[N].append(dfgrid2['Test_val'][ind_r])
            prob_re[N].append(dfgrid2['Avg_prob_test'][ind_r])
        ind_s = np.absolute(np.mean(np.vstack(dfgrid['Avg_prob_test']),axis = 1)-etas[i]).argmin()
        val_st_lower[N].append(dfgrid['Lower_test'][ind_s])
        val_st_upper[N].append(dfgrid['Upper_test'][ind_s])
        val_st[N].append(dfgrid['Test_val'][ind_s])
        prob_st[N].append(dfgrid['Avg_prob_test'][ind_s])
        ind_r = np.absolute(np.mean(np.vstack(dfgrid2['Avg_prob_test']),axis = 1)-etas[i]).argmin()
        ind_2 = np.absolute(np.mean(np.vstack(dfgrid2['Eps']),axis = 1)-1).argmin()
        val_re_lower[N].append(dfgrid2['Lower_test'][ind_r])
        val_re_upper[N].append(dfgrid2['Upper_test'][ind_r])
        val_re[N].append(dfgrid2['Test_val'][ind_r])
        prob_re[N].append(dfgrid2['Avg_prob_test'][ind_r])
        val_re_nom_upper[N].append(dfgrid2['Upper_test'][ind_2])
        val_re_nom_lower[N].append(dfgrid2['Lower_test'][ind_2])
        val_re_nom[N].append(dfgrid2['Test_val'][ind_2])
        prob_re_nom[N].append(dfgrid2['Avg_prob_test'][ind_2])

    plt.figure(figsize = (8,3))
    plt.plot(prob_st[N], val_st[N], label = "Mean-Var", color = "tab:blue")
    # plt.plot(prob_re[N], val_re[N], label = "Reshaped", color = "tab:orange")
    plt.fill_between(prob_st[N],val_st_lower[N],val_st_upper[N], color = "tab:blue", alpha=0.3)
    # plt.fill_between(prob_re[N],val_re_lower[N],val_re_upper[N], color = "tab:orange", alpha=0.3)
    plt.plot(prob_re_nom[N],val_re_nom[N],label="Reshaped_orig", color = "tab:green")
    plt.fill_between(prob_re_nom[N],val_re_nom_lower[N],val_re_nom_upper[N], color = "tab:green", alpha=0.3)
    plt.xlabel("Prob. of cons. vio.")
    plt.ylabel("Objective value")
    plt.title(f"N={N}")
    plt.legend()
    plt.savefig(foldername + f"{N}", bbox_inches='tight')
    plt.show()

