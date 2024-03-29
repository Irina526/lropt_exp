import argparse
import os
import sys
import joblib
from joblib import Parallel, delayed
output_stream = sys.stdout

import cvxpy as cp
import scipy as sc
import numpy as np
import numpy.random as npr
import torch
from sklearn import datasets
import pandas as pd
import lropt
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
R = 2
n = 20


def plot_iters(dftrain, dftest, title, steps=2000, logscale=True):
    plt.rcParams.update({
        "text.usetex": True,

        "font.size": 22,
        "font.family": "serif"
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 3))
    len_train = len(dftrain["Violations_train"])
    ax1.plot(np.arange(0,len_train,10), dftest["Violations_test"][:steps],
             label="Out-of-sample empirical CVaR")
    ax1.plot(dftrain["Violations_train"][:steps],
             label="In-sample empirical CVaR", linestyle="--")

    ax1.set_xlabel("Iterations")
    ax1.hlines(xmin=0, xmax=dftrain["Violations_train"][:steps].shape[0],
               y=-0.0, linestyles="--", color="black", label="Target threshold: 0")
    ax1.legend()
    ax2.plot(dftest["Test_val"][:steps], label="Objective value")
    ax2.set_xlabel("Iterations")
    ax2.ticklabel_format(style="sci", axis='y',
                         scilimits=(0, 0), useMathText=True)
    ax2.legend()
    if logscale:
        ax1.set_xscale("log")
        ax2.set_xscale("log")
    plt.savefig(title+"_iters", bbox_inches='tight')

def plot_coverage_all(df_standard,df_reshape,dfs,title,title1,ind_1 = (0,100), ind_2 = (0,100), logscale = True, legend = False, zoom = False):
    plt.rcParams.update({
    "text.usetex":True,

    "font.size":22,
    "font.family": "serif"
})
    beg1,end1 = ind_1
    beg2,end2 = ind_2

    fig, (ax, ax1,ax2) = plt.subplots(1, 3, figsize=(23, 3))
    
    ax.plot(df_standard['Probability_violations_test'][beg1:end1], df_standard['Test_val'][beg1:end1], color="tab:blue", label=r"Mean-Var set")
    ax.fill(np.append(df_standard['Probability_violations_test'][beg1:end1],df_standard['Probability_violations_test'][beg1:end1][::-1]), np.append(df_standard['Lower_test'][beg1:end1],df_standard['Upper_test'][beg1:end1][::-1]), color="tab:blue", alpha=0.2)

    ax.plot(df_reshape['Probability_violations_test'][beg2:end2], df_reshape['Test_val'][beg2:end2], color="tab:orange", label=r"Reshaped set")
    ax.fill(np.append(df_reshape['Probability_violations_test'][beg2:end2],df_reshape['Probability_violations_test'][beg2:end2][::-1]), np.append(df_reshape['Lower_test'][beg2:end2],df_reshape['Upper_test'][beg2:end2][::-1]), color="tab:orange", alpha=0.2)
    ax.set_xlabel("Probability of constraint violation")
    ax.axvline(x = 0.03, color = "green", linestyle = "-.",label = r"$\eta = 0.03$")
    ax.set_ylabel("Objective value")
    ax.set_title(title1)
    # ax.set_yticks(ticks = [-2e1,0,2e1])
    # ax.set_yticks(ticks = [-1,0,1])
    # ax.set_ylim([])
    ax.ticklabel_format(style="sci",axis='y',scilimits = (0,0), useMathText=True)
    # ax.legend()

    ax1.plot(np.mean(np.vstack(df_standard['Coverage_test']),axis = 1)[beg1:end1], np.mean(np.vstack(df_standard['Test_val']),axis = 1)[beg1:end1], color="tab:blue", label=r"Mean-Var set")
    ax1.fill(np.append(np.quantile(np.vstack(df_standard['Coverage_test']),0.1,axis = 1)[beg1:end1],np.quantile(np.vstack(df_standard['Coverage_test']),0.9,axis = 1)[beg1:end1][::-1]), np.append(np.quantile(np.vstack(df_standard['Test_val']),0.1,axis = 1)[beg1:end1],np.quantile(np.vstack(df_standard['Test_val']),0.90,axis = 1)[beg1:end1][::-1]), color="tab:blue", alpha=0.2)

    ax1.plot(np.mean(np.vstack(df_reshape['Coverage_test']),axis = 1)[beg2:end2],np.mean(np.vstack(df_reshape['Test_val']),axis = 1)[beg2:end2], color = "tab:orange",label=r"Decision-Focused set")
    ax1.fill(np.append(np.quantile(np.vstack(df_reshape['Coverage_test']),0.1,axis = 1)[beg2:end2],np.quantile(np.vstack(df_reshape['Coverage_test']),0.9,axis = 1)[beg2:end2][::-1]), np.append(np.quantile(np.vstack(df_reshape['Test_val']),0.1,axis = 1)[beg2:end2],np.quantile(np.vstack(df_reshape['Test_val']),0.90,axis = 1)[beg2:end2][::-1]), color="tab:orange", alpha=0.2)
    if dfs:
        for i in range(5):
            ax1.plot(np.mean(np.vstack(dfs[i+1][0]['Coverage_test']),axis = 1)[beg1:end1], np.mean(np.vstack(dfs[i+1][0]['Test_val']),axis = 1)[beg1:end1], color="tab:blue", linestyle = "-")
            ax1.plot(np.mean(np.vstack(dfs[i+1][1]['Coverage_test']),axis = 1)[beg2:end2],np.mean(np.vstack(dfs[i+1][1]['Test_val']),axis = 1)[beg2:end2], color = "tab:orange",linestyle = "-")

    ax1.ticklabel_format(style="sci",axis='y',scilimits = (0,0), useMathText=True)
    ax1.axvline(x = 0.8, color = "black", linestyle = ":",label = "0.8 Coverage")

    if logscale:
        ax1.set_xscale("log")
    # ax1.set_yticks(ticks = [-1,0,1])

    ax1.set_xlabel("Test set coverage")
    ax1.set_ylabel("Objective value")
    # ax1.legend()

    ax2.plot(df_standard['Coverage_test'][beg1:end1], df_standard['Probability_violations_test'][beg1:end1], color="tab:blue", label=r"Mean-Var set")

    ax2.plot(df_reshape['Coverage_test'][beg2:end2], df_reshape['Probability_violations_test'][beg2:end2], color="tab:orange", label=r"Reshaped set",alpha = 0.8)
    if dfs:
        for i in range(5):
            ax2.plot(np.mean(np.vstack(dfs[i+1][0]['Coverage_test']),axis = 1)[beg1:end1], np.mean(np.vstack(dfs[i+1][0]['Probability_violations_test']),axis = 1)[beg1:end1], color="tab:blue", linestyle = "-")
            ax2.plot(np.mean(np.vstack(dfs[i+1][1]['Coverage_test']),axis = 1)[beg2:end2],np.mean(np.vstack(dfs[i+1][1]['Probability_violations_test']),axis = 1)[beg2:end2], color = "tab:orange",linestyle = "-")
    # ax2.plot(np.arange(100)/100, 1 - np.arange(100)/100, color = "red")
    # ax2.set_ylim([-0.05,0.25])
    ax2.axvline(x = 0.8, color = "black",linestyle = ":", label = "0.8 Coverage")
    ax2.axhline(y = 0.03, color = "green",linestyle = "-.", label = r"$\hat{\eta} = 0.03$")
    ax2.set_ylabel("Prob. of cons. vio.")
    ax2.set_xlabel("Test set coverage")
    if zoom:
        axins = zoomed_inset_axes(ax2, 6, loc="upper center")
        axins.set_xlim(-0.005, 0.1)
        axins.set_ylim(-0.001,0.035)
        axins.plot(np.mean(np.vstack(df_standard['Coverage_test']),axis = 1)[beg1:end1], np.mean(np.vstack(df_standard['Probability_violations_test']),axis = 1)[beg1:end1], color="tab:blue")
        axins.plot(np.mean(np.vstack(df_reshape['Coverage_test']),axis = 1)[beg2:end2], np.mean(np.vstack(df_reshape['Probability_violations_test']),axis = 1)[beg2:end2], color="tab:orange",alpha = 0.8)
        axins.axhline(y = 0.03, color = "green",linestyle = "-.", label = r"$\hat{\eta} = 0.03$")
        axins.set_xticks(ticks=[])
        axins.set_yticks(ticks=[])
        mark_inset(ax2, axins, loc1=3, loc2=4, fc="none", ec="0.5")
    if logscale:
        ax2.set_xscale("log")
    if legend:
        ax2.legend(bbox_to_anchor=(-1.8, -0.6, 0, 0), loc="lower left",
                 borderaxespad=0, ncol=4, fontsize = 24)
    plt.subplots_adjust(left=0.1)
    plt.savefig(title+"_curves",bbox_inches='tight')
    # plt.show()


val_st = []
val_re = []
prob_st = []
prob_re = []
nvals = np.array([50,100,500,1000,1500,2000,3000,4000,5000])
for N in nvals:
    dfgrid = pd.read_csv(foldername +f"gridmv_{N,n,0}.csv")
    dfgrid = dfgrid.drop(columns=["step"])
    dfgrid2 = pd.read_csv(foldername +f"gridre_{N,n,0}.csv")
    dfgrid2 = dfgrid2.drop(columns=["step"])
    df_test = pd.read_csv(foldername +f"trainval_{N,n,0}.csv")
    df = pd.read_csv(foldername +f"train_{N,n,0}.csv")
    # df_test.drop(columns=["step"])
    # df.drop(columns=["step"])
    for r in range(1,R):
        newgrid = pd.read_csv(foldername +f"gridmv_{N,n,r}.csv")
        newgrid = newgrid.drop(columns=["step"])
        dfgrid = dfgrid.add(newgrid.reset_index(), fill_value=0)
        newgrid2 = pd.read_csv(foldername +f"gridre_{N,n,r}.csv")
        newgrid2 = newgrid2.drop(columns=["step"])
        dfgrid2 = dfgrid2.add(newgrid2.reset_index(), fill_value=0)
        # newdf_test = pd.read_csv(foldername +f"trainval_{N,n,r}.csv")
        # df_test = df_test.add(newdf_test.reset_index(), fill_value=0)
        # newdf = pd.read_csv(foldername +f"train_{N,n,r}.csv")
        # df = df.add(newdf.reset_index(), fill_value=0)

    if R > 1:
        dfgrid = dfgrid/R
        dfgrid2 = dfgrid2/R
        # df_test = df_test/R
        # df = df/R
        dfgrid.to_csv(foldername + f"results/gridmv_{N,n}.csv")
        dfgrid2.to_csv(foldername +f"results/gridre_{N,n}.csv")
        # df_test.to_csv(foldername +f"results/trainval_{N,n}.csv")
        # df.to_csv(foldername +f"results/train_{N,n}.csv")

        plot_coverage_all(dfgrid,dfgrid2,None, foldername + f"results/port(N,m,r)_{N,n}", f"port(N,m,r)_{N,n,r}", ind_1=(0,10000),ind_2=(0,10000), logscale = False, zoom = False,legend = True)

        # plot_iters(df, df_test, foldername + f"results/port(N,m)_{N,n}", steps = 10000,logscale = 1)

    ind_s = np.absolute(np.mean(np.vstack(dfgrid['Probability_violations_test']),axis = 1)-0).argmin()
    val_st.append(dfgrid['Test_val'][ind_s])
    prob_st.append(dfgrid['Probability_violations_test'][ind_s])

    ind_r = np.absolute(np.mean(np.vstack(dfgrid2['Probability_violations_test']),axis = 1)-0).argmin()
    val_re.append(dfgrid2['Test_val'][ind_r])
    prob_re.append(dfgrid2['Probability_violations_test'][ind_r])


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 3))
ax1.plot(nvals,val_st, label = "Mean-Var")
ax1.plot(nvals, val_re, label = "Reshaped")
ax1.set_xlabel("Number of Samples")
ax1.set_title(f"m:{n} OOS Test Value")
ax1.legend()

ax2.plot(nvals,prob_st, label = "Mean-Var")
ax2.plot(nvals, prob_re, label = "Reshaped")
ax2.set_xlabel("Number of Samples")
ax2.set_title(f"m:{n} OOS Prob Violations")
ax2.legend()
plt.savefig(foldername + f"results/m:{n}_varyN",bbox_inches='tight')