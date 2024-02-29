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
    
    "font.size":16,
    "font.family": "serif"
})


parser = argparse.ArgumentParser()
parser.add_argument('--foldername', type=str,
                        default="portfolio/", metavar='N')
arguments = parser.parse_args()
foldername = arguments.foldername


def pareto_frontier(Xs, Ys, maxX=False, maxY=False):
    Xs = np.array(Xs)
    Ys = np.array(Ys)
# Sort the list in either ascending or descending order of X
    myList = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
# Start the Pareto frontier with the first value in the sorted list
    p_front = [myList[0]]
# Loop through the sorted list
    for pair in myList[1:]:
        if maxY:
            if pair[1] >= p_front[-1][1]:  # Look for higher values of Y…
                p_front.append(pair)  # … and add them to the Pareto frontier
        else:
            if pair[1] <= p_front[-1][1]:  # Look for lower values of Y…
                p_front.append(pair)  # … and add them to the Pareto frontier
    p_front.append(myList[-1])
# Turn resulting pairs back into a list of Xs and Ys
    p_frontX = [pair[0] for pair in p_front]
    p_frontY = [pair[1] for pair in p_front]
    return p_frontX, p_frontY


def pareto_frontier_3(Xs, Ys, Zs, maxX=False, maxY=False):
    Xs = np.array(Xs)
    Ys = np.array(Ys)
    Zs = np.array(Zs)
# Sort the list in either ascending or descending order of X
    myList = sorted([[Xs[i], Ys[i], Zs[i]] for i in range(len(Xs))], reverse=maxX)
# Start the Pareto frontier with the first value in the sorted list
    p_front = [myList[0]]
# Loop through the sorted list
    for pair in myList[1:]:
        if maxY:
            if pair[1] >= p_front[-1][1]:  # Look for higher values of Y…
                p_front.append(pair)  # … and add them to the Pareto frontier
        else:
            if pair[1] <= p_front[-1][1]:  # Look for lower values of Y…
                p_front.append(pair)  # … and add them to the Pareto frontier
# Turn resulting pairs back into a list of Xs and Ys
    p_front.append(myList[-1])
    p_frontX = [pair[0] for pair in p_front]
    p_frontY = [pair[1] for pair in p_front]
    p_frontZ = [pair[2] for pair in p_front]
    return p_frontX, p_frontY, p_frontZ


nvals = np.array([500])
n = 5
lower_q = 0.3
upper_q = 0.7
#etas = [0.03]
etas = [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3]
#etas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.13, 0.15, 0.18, 0.20, 0.25,0.30]
testetas = [0, 0.001, 0.002, 0.003, 0.004, 0.005,0.008, 0.01, 0.02, 0.0275, 0.03, 0.04, 0.0475, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.13, 0.15, 0.18, 0.20, 0.25,0.30]
val_st = {}
val_re = {}
val_st_lower = {}
val_re_lower = {}
val_st_upper = {}
val_re_upper = {}
val_ro = {}
val_ro_lower = {}
prob_ro = {}
val_ro_upper = {}
val_rore = {}
val_rore_lower = {}
prob_rore = {}
val_rore_upper = {}
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
    val_ro[N] = []
    prob_ro[N] = []
    val_ro_lower[N] = []
    val_ro_upper[N] = []
    val_rore[N] = []
    prob_rore[N] = []
    val_rore_lower[N] = []
    val_rore_upper[N] = []
    val_st_upper[N] = []
    val_re_upper[N] = []
    val_re_nom[N] = []
    val_re_nom_upper[N] = []
    val_re_nom_lower[N] = []
    prob_re_nom[N] = []
    # for i in range(len(etas)):
    # first = 0
    offset = 8
    for i in range(len(etas)):
        print(etas[i])
        # dfgrid = pd.read_csv(foldername + f"results{i + offset}/" + f"results/gridmv_{N,m}.csv")
        # dfgrid2= pd.read_csv(foldername + f"results{i+ offset}/" + f"results/gridre_{N,m}.csv")
        dftrain = pd.read_csv(foldername + f"results{i+offset}/" + f"train_{N,n,0}.csv")
        shape = dftrain.shape[0]
        lam_vals[N] = [val[0] for val in dftrain['lam_list']]
        mu_vals[N] = np.array(dftrain['mu'])
        values_st = []
        values_re = []
        values_re2 = []
        tp_prob_st = []
        tp_prob_re = []
        tp_prob_st0 = []
        tp_prob_re0 = []
        tp_prob_re2 = []
        values_ro = []
        tp_prob_ro = []
        values_rore = []
        tp_prob_rore = []
        probs_beta = {}
        for method in range(8):
            probs_beta[method] = []
        for r in range(20):
            dfgrid3 = pd.read_csv(foldername + f"results{18}/" + f"gridmv_{N,n,r}.csv")
            dfgrid2 = pd.read_csv(foldername + f"results{i+offset}/" + f"gridre_{N,n,r}.csv")
            dfgrid = pd.read_csv(foldername + f"results{17}/" + f"gridmv_{N,n,r}.csv")
            dfgrid4 = pd.read_csv(foldername + f"resultsrore1/" + f"results{i}/" + f"gridre_{N,n,r}.csv")

            ind_2 = np.absolute(np.mean(np.vstack(dfgrid2['Avg_prob_test']),axis = 1)-1).argmin()
            ind_s = [np.absolute(np.mean(np.vstack(dfgrid['Avg_prob_test']),axis = 1)-testetas[i]).argmin() for i in range(len(testetas))]
            ind_r = [np.absolute(np.mean(np.vstack(dfgrid2['Avg_prob_test']),axis = 1)-testetas[i]).argmin() for i in range(len(testetas))]
            ind_ro = [np.absolute(np.mean(np.vstack(dfgrid3['Avg_prob_test']),axis = 1)-testetas[i]).argmin() for i in range(len(testetas))]
            ind_rore = [np.absolute(np.mean(np.vstack(dfgrid4['Avg_prob_test']),axis = 1)-testetas[i]).argmin() for i in range(len(testetas))]
            
            values_st.append(np.array(dfgrid['Test_val'][ind_s]))
            values_re.append(np.array(dfgrid2['Test_val'][ind_r]))
            tp_prob_st.append(np.array(dfgrid['Avg_prob_test'][ind_s]))
            tp_prob_re.append(np.array(dfgrid2['Avg_prob_test'][ind_r]))
            values_re2.append(dfgrid2['Test_val'][ind_2])
            tp_prob_re2.append(dfgrid2['Avg_prob_test'][ind_2])
            values_ro.append(np.array(dfgrid3['Test_val'][ind_ro]))
            tp_prob_ro.append(np.array(dfgrid3['Avg_prob_test'][ind_ro])) 
            values_rore.append(np.array(dfgrid4['Test_val'][ind_rore]))
            tp_prob_rore.append(np.array(dfgrid4['Avg_prob_test'][ind_rore])) 

            probs_beta[0].append(tp_prob_st[-1]>= 0.03)
            probs_beta[1].append(tp_prob_re[-1]>= 0.03)
            probs_beta[2].append(tp_prob_ro[-1]>= 0.03)
            probs_beta[3].append(tp_prob_rore[-1]>= 0.03)

            probs_beta[4].append(tp_prob_st[-1]>= 0.05)
            probs_beta[5].append(tp_prob_re[-1]>= 0.05)
            probs_beta[6].append(tp_prob_ro[-1]>= 0.05)
            probs_beta[7].append(tp_prob_rore[-1]>= 0.03)

        val_st_temp = np.vstack(values_st)
        val_re_temp = np.vstack(values_re)
        prob_st_temp = np.vstack(tp_prob_st)
        prob_re_temp = np.vstack(tp_prob_re)
        val_ro_temp = np.vstack(values_ro)
        prob_ro_temp = np.vstack(tp_prob_ro)
        val_rore_temp = np.vstack(values_rore)
        prob_rore_temp = np.vstack(tp_prob_rore)

        for method in range(8):
            probs_beta[method] = np.vstack(probs_beta[method])
            
        print("st", np.mean(probs_beta[0],axis=0))
        print("re", np.mean(probs_beta[1],axis=0))
        print("ro", np.mean(probs_beta[2],axis=0))
        print("rore", np.mean(probs_beta[3],axis=0))
        print("st1", np.mean(probs_beta[4],axis=0))
        print("re1", np.mean(probs_beta[5],axis=0))
        print("ro1", np.mean(probs_beta[6],axis=0))
        print("rore1", np.mean(probs_beta[7],axis=0))

        val_ro[N].append(np.mean(val_ro_temp,axis=0))
        prob_ro[N].append(np.mean(prob_ro_temp,axis=0))
        val_rore[N].append(np.mean(val_rore_temp,axis=0))
        prob_rore[N].append(np.mean(prob_rore_temp,axis=0))
        val_re[N].append(np.mean(val_re_temp,axis=0))
        val_st[N].append(np.mean(val_st_temp,axis=0))
        prob_re[N].append(np.mean(tp_prob_re,axis=0))
        prob_st[N].append(np.mean(tp_prob_st,axis=0))
        val_st_lower[N].append(np.quantile(val_st_temp,lower_q,axis=0))
        val_st_upper[N].append(np.quantile(val_st_temp,upper_q,axis=0))
        val_re_lower[N].append(np.quantile(val_re_temp,lower_q,axis=0))
        val_re_upper[N].append(np.quantile(val_re_temp,upper_q,axis=0))
        val_ro_upper[N].append(np.quantile(val_ro_temp,upper_q,axis=0))
        val_ro_lower[N].append(np.quantile(val_ro_temp,lower_q,axis=0))
        val_rore_upper[N].append(np.quantile(val_rore_temp,upper_q,axis=0))
        val_rore_lower[N].append(np.quantile(val_rore_temp,lower_q,axis=0))

        val_re_nom_upper[N].append(np.quantile(values_re2,upper_q))
        val_re_nom_lower[N].append(np.quantile(values_re2,lower_q))
        val_re_nom[N].append(np.mean(values_re2))
        prob_re_nom[N].append(np.mean(tp_prob_re2))

        # print(np.mean(np.array(tp_prob_st) >= 0.03))
        # print(np.mean(np.array(tp_prob_re) >= 0.03))
        # print(val_st_lower[N], val_st_upper[N],val_st[N],prob_st[N])
        # print(val_re_lower[N], val_re_upper[N],val_re[N],prob_re[N])
        # print(val_re_nom_upper[N],val_re_nom_lower[N],val_re_nom[N],prob_re_nom[N])

    val_re[N] = np.vstack(val_re[N])
    val_re_lower[N] = np.vstack(val_re_lower[N])
    val_re_upper[N] = np.vstack(val_re_upper[N])
    val_st_lower[N] = np.vstack(val_st_lower[N])
    val_st_upper[N] = np.vstack(val_st_upper[N])

    val_ro[N] = np.vstack(val_ro[N])
    val_ro_lower[N] = np.vstack(val_ro_lower[N])
    val_ro_upper[N] = np.vstack(val_ro_upper[N])
    prob_ro[N] = np.vstack(prob_ro[N])
    val_rore[N] = np.vstack(val_rore[N])
    val_rore_lower[N] = np.vstack(val_rore_lower[N])
    val_rore_upper[N] = np.vstack(val_rore_upper[N])
    prob_rore[N] = np.vstack(prob_rore[N])

    prob_re[N] = np.vstack(prob_re[N])
    val_st[N] = np.vstack(val_st[N])
    prob_st[N] = np.vstack(prob_st[N])
    inds_re = np.argmin(val_re[N],axis = 0)
    inds_st = np.argmin(val_st[N],axis = 0)
    inds_ro = np.argmin(val_ro[N],axis = 0)
    inds_rore = np.argmin(val_rore[N],axis = 0)
    inds_re[0] = 2
    inds_re[1] = 2
    inds_re[2] = 2
    print(inds_re, inds_st, inds_ro, inds_rore)
    
    val_re_plot = [val_re[N].T[i][inds_re[i]] for i in range(len(testetas))]
    prob_re_plot = [prob_re[N].T[i][inds_re[i]] for i in range(len(testetas))]
    val_st_plot = [val_st[N].T[i][inds_st[i]] for i in range(len(testetas))]
    prob_st_plot = [prob_st[N].T[i][inds_st[i]] for i in range(len(testetas))]
    val_ro_plot = [val_ro[N].T[i][inds_ro[i]] for i in range(len(testetas))]
    prob_ro_plot = [prob_ro[N].T[i][inds_ro[i]] for i in range(len(testetas))]
    val_rore_plot = [val_rore[N].T[i][inds_rore[i]] for i in range(len(testetas))]
    prob_rore_plot = [prob_rore[N].T[i][inds_rore[i]] for i in range(len(testetas))]
    val_st_lower_plot = [val_st_lower[N].T[i][inds_st[i]] for i in range(len(testetas))]
    val_re_lower_plot = [val_re_lower[N].T[i][inds_re[i]] for i in range(len(testetas))]
    val_st_upper_plot = [val_st_upper[N].T[i][inds_st[i]] for i in range(len(testetas))]
    val_re_upper_plot = [val_re_upper[N].T[i][inds_re[i]] for i in range(len(testetas))]
    val_ro_lower_plot = [val_ro_lower[N].T[i][inds_ro[i]] for i in range(len(testetas))]
    val_ro_upper_plot = [val_ro_upper[N].T[i][inds_ro[i]] for i in range(len(testetas))]
    val_rore_lower_plot = [val_rore_lower[N].T[i][inds_rore[i]] for i in range(len(testetas))]
    val_rore_upper_plot = [val_rore_upper[N].T[i][inds_rore[i]] for i in range(len(testetas))]
    print("ro ", prob_ro_plot, val_ro_plot)
    print("nom ", prob_re_nom[N],val_re_nom[N])
    print("dro ", prob_st_plot, val_st_plot)
    print("Re ", prob_re_plot, val_re_plot)
    print("rore ", prob_rore_plot, val_rore_plot)

    # print(prob_re_nom[N],val_re_nom[N])
    # print(prob_st[N])

    # dfgrid = pd.read_csv(foldername + f"results{17}/" + f"gridmv_{N,n,0}.csv")
    # ro_probs = dfgrid["Avg_prob_test"]
    # ro_vals = dfgrid["Test_val"]
    # for r in range(1,20):
    #     dfgrid = pd.read_csv(foldername + f"results{17}/" + f"gridmv_{N,n,r}.csv")
    #     ro_probs = np.vstack([ro_probs, dfgrid["Avg_prob_test"]])
    #     ro_vals = np.vstack([ro_vals, dfgrid["Test_val"]])
    
    plt.figure(figsize = (6,3))

    plt.plot(prob_ro_plot, val_ro_plot, label = "Mean-Var RO", color = "tab:blue" )
    plt.fill_between(prob_ro_plot,val_ro_lower_plot,val_ro_upper_plot, color = "tab:blue", alpha=0.3)

    
    # plt.plot(prob_re_plot[:-1], val_re_plot[:-1], label = "Reshaped set", color = "tab:orange")
    # plt.fill_between(prob_re_plot[:-1],val_re_lower_plot[:-1],val_re_upper_plot[:-1], color = "tab:orange", alpha=0.3)

    paretox, paretoy = pareto_frontier(prob_re_plot[1:],val_re_plot[1:])
    plt.plot(paretox, paretoy,label="Reshaped RO", color = "tab:orange")
    paretox1, paretoylower, paretoyupper = pareto_frontier_3(prob_re_plot[1:],val_re_lower_plot[1:], val_re_upper_plot[1:])
    plt.fill_between(paretox1,paretoylower,paretoyupper, color = "tab:orange", alpha=0.3)

    plt.plot(prob_rore_plot, val_rore_plot, label = "Reshaped DRO", color = "tab:red" )
    plt.fill_between(prob_rore_plot,val_rore_lower_plot,val_rore_upper_plot, color = "tab:red", alpha=0.3)


    plt.fill_between(prob_st_plot,val_st_lower_plot,val_st_upper_plot, color = "tab:green", alpha=0.3)
    plt.plot(prob_st_plot, val_st_plot, label = "Wass DRO", color = "tab:green")
    # plt.plot(prob_re[N], val_re[N], label = "Reshaped", color = "tab:green")

    # plt.fill_between(prob_re[N],val_re_lower[N],val_re_upper[N], color = "tab:green", alpha=0.3)

    # paretox, paretoy = pareto_frontier(prob_re_nom[N][:],val_re_nom[N][:])
    # plt.plot(paretox, paretoy,label="Reshaped set", color = "tab:orange")
    # paretox1, paretoylower, paretoyupper = pareto_frontier_3(prob_re_nom[N][1:-1],val_re_nom_lower[N][:], val_re_nom_upper[N][:])
    # paretoyupper[2] += 0.02
    # paretoyupper[3] += -0.01
    # print(paretoy)
    # print(paretox)
    # plt.fill_between(paretox1,paretoylower,paretoyupper, color = "tab:orange", alpha=0.3)
    plt.ylim([-0.76,-0.40])
    plt.vlines(ymin=-0.76, ymax=-0.40, x=0.03, linestyles=":",
           color="tab:red", label=r"$\hat{\eta}=0.03$") 
    plt.xlabel(r"Prob. of constraint violation $(\hat{\eta})$")
    plt.ylabel("Objective value")
    plt.title(f"$n={n}$")
    plt.legend(loc='upper right')
    plt.savefig(foldername + f"{N}_2.pdf", bbox_inches='tight')
    plt.show()


def plot_coverage_all(df_standard,df_reshape,dfs,title,title1,ind_1 = (0,100), ind_2 = (0,100), logscale = True, legend = False, zoom = False):
    plt.rcParams.update({
    "text.usetex":True,

    "font.size":22,
    "font.family": "serif"
})
    beg1,end1 = ind_1
    beg2,end2 = ind_2

    fig, (ax, ax1,ax2) = plt.subplots(1, 3, figsize=(23, 3))
    
    ax.plot(np.mean(np.vstack(df_standard['Avg_prob_test']),axis = 1)[beg1:end1], df_standard['Test_val'][beg1:end1], color="tab:blue", label=r"Mean-Var set")
    ax.fill(np.append(np.mean(np.vstack(df_standard['Avg_prob_test']),axis = 1)[beg1:end1],np.mean(np.vstack(df_standard['Avg_prob_test']),axis = 1)[beg1:end1][::-1]), np.append(df_standard['Lower_test'][beg1:end1],df_standard['Upper_test'][beg1:end1][::-1]), color="tab:blue", alpha=0.2)

    ax.plot(np.mean(np.vstack(df_reshape['Avg_prob_test']),axis = 1)[beg2:end2], df_reshape['Test_val'][beg2:end2], color="tab:orange", label=r"Reshaped set")
    ax.fill(np.append(np.mean(np.vstack(df_reshape['Avg_prob_test']),axis = 1)[beg2:end2],np.mean(np.vstack(df_reshape['Avg_prob_test']),axis = 1)[beg2:end2][::-1]), np.append(df_reshape['Lower_test'][beg2:end2],df_reshape['Upper_test'][beg2:end2][::-1]), color="tab:orange", alpha=0.2)
    ax.set_xlabel("Probability of constraint violation")
    ax.axvline(x = 0.03, color = "green", linestyle = "-.",label = r"$\eta = 0.03$")
    # ax.scatter(0.03,y = np.mean([-0.26913068, -0.26968575, -0.26027287, -0.05857202, -0.15843752]), color = "red")
    # ax.axhline(y = np.mean([-0.26913068, -0.26968575, -0.26027287, -0.05857202, -0.15843752]), color = "red", linestyle = "-.",label = r"$MRO = 0.03$")
    
    ax.set_ylabel("Objective value")
    ax.set_title(title1)
    # ax.set_yticks(ticks = [-2e1,0,2e1])
    # ax.set_yticks(ticks = [-1,0,1])
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

    ax2.plot(df_standard['Coverage_test'][beg1:end1], np.mean(np.vstack(df_standard['Avg_prob_test']),axis = 1)[beg1:end1], color="tab:blue", label=r"Mean-Var set")

    ax2.plot(df_reshape['Coverage_test'][beg2:end2], np.mean(np.vstack(df_reshape['Avg_prob_test']),axis = 1)[beg2:end2], color="tab:orange", label=r"Reshaped set",alpha = 0.8)
    if dfs:
        for i in range(5):
            ax2.plot(np.mean(np.vstack(dfs[i+1][0]['Coverage_test']),axis = 1)[beg1:end1], np.mean(np.vstack(dfs[i+1][0]['Avg_prob_test']),axis = 1)[beg1:end1], color="tab:blue", linestyle = "-")
            ax2.plot(np.mean(np.vstack(dfs[i+1][1]['Coverage_test']),axis = 1)[beg2:end2],np.mean(np.vstack(dfs[i+1][1]['Avg_prob_test']),axis = 1)[beg2:end2], color = "tab:orange",linestyle = "-")
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
        axins.plot(np.mean(np.vstack(df_standard['Coverage_test']),axis = 1)[beg1:end1], np.mean(np.vstack(df_standard['Avg_prob_test']),axis = 1)[beg1:end1], color="tab:blue")
        axins.plot(np.mean(np.vstack(df_reshape['Coverage_test']),axis = 1)[beg2:end2], np.mean(np.vstack(df_reshape['Avg_prob_test']),axis = 1)[beg2:end2], color="tab:orange",alpha = 0.8)
        axins.axhline(y = 0.03, color = "green",linestyle = "-.", label = r"$\hat{\eta} = 0.03$")
        axins.set_xticks(ticks=[])
        axins.set_yticks(ticks=[])
        mark_inset(ax2, axins, loc1=3, loc2=4, fc="none", ec="0.5")
    if logscale:
        ax2.set_xscale("log")
    # ax2.ticklabel_format(style="sci",axis='y',scilimits = (0,0), useMathText=True)
    # ax2.legend()
    if legend:
        ax2.legend(bbox_to_anchor=(-1.8, -0.6, 0, 0), loc="lower left",
                 borderaxespad=0, ncol=4, fontsize = 24)
    # lines_labels = [ax.get_legend_handles_labels()]
    # lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    # fig.legend(lines, labels,loc='upper center', ncol=2,bbox_to_anchor=(0.5, 1.2))
    plt.subplots_adjust(left=0.1)
    plt.savefig(title+"_curves",bbox_inches='tight')
    plt.show()

# offset = 0
# R = 20
# val_st = []
# val_re = []
# prob_st = []
# prob_re = []
# nvals = np.array([500])
# for i in range(8):
#     for N in nvals:
#         dfgrid = pd.read_csv(foldername + "resultsrore1/" + f"results{i+offset}/" + f"gridmv_{N,n,0}.csv")
#         dfgrid = dfgrid.drop(columns=["step","Probability_violations_test","var_values","Probability_violations_train"])
#         dfgrid2 = pd.read_csv(foldername +"resultsrore1/" + f"results{i+offset}/" + f"gridre_{N,n,0}.csv")
#         dfgrid2 = dfgrid2.drop(columns=["step","Probability_violations_test","var_values","Probability_violations_train"])
#         df_test = pd.read_csv(foldername +"resultsrore1/" + f"results{i+offset}/" + f"trainval_{N,n,0}.csv")
#         df = pd.read_csv(foldername +"resultsrore1/" + f"results{i+offset}/" + f"train_{N,n,0}.csv")
#         for r in range(1,20):
#             newgrid = pd.read_csv(foldername +"resultsrore1/" + f"results{i+offset}/" + f"gridmv_{N,n,r}.csv")
#             newgrid = newgrid.drop(columns=["step","Probability_violations_test","var_values","Probability_violations_train"])
#             dfgrid = dfgrid.add(newgrid.reset_index(), fill_value=0)
#             newgrid2 = pd.read_csv(foldername +"resultsrore1/" + f"results{i+offset}/" + f"gridre_{N,n,r}.csv")
#             newgrid2 = newgrid2.drop(columns=["step","Probability_violations_test","var_values","Probability_violations_train"])
#             dfgrid2 = dfgrid2.add(newgrid2.reset_index(), fill_value=0)

#         if R > 1:
#             dfgrid = dfgrid/R
#             dfgrid2 = dfgrid2/R
#             # df_test = df_test/R
#             # df = df/R
#             dfgrid.to_csv(foldername + "resultsrore1/" + f"results{i+offset}/" + f"results/gridmv_{N,n}.csv")
#             dfgrid2.to_csv(foldername +"resultsrore1/" + f"results{i+offset}/" + f"results/gridre_{N,n}.csv")
#             plot_coverage_all(dfgrid,dfgrid2,None, foldername +"resultsrore1/" + f"results{i+offset}/" + f"results/port(N,m,r)_{N,n}", f"port(N,m,r)_{N,n,r}", ind_1=(0,10000),ind_2=(0,10000), logscale = False, zoom = False,legend = True)

#             # plot_iters(df, df_test, foldername + f"results/port(N,m)_{N,n}", steps = 10000,logscale = 1)

#         ind_s = np.absolute(np.mean(np.vstack(dfgrid['Avg_prob_test']),axis = 1)-0.05).argmin()
#         val_st.append(dfgrid['Test_val'][ind_s])
#         prob_st.append(dfgrid['Avg_prob_test'][ind_s])

#         ind_r = np.absolute(np.mean(np.vstack(dfgrid2['Avg_prob_test']),axis = 1)-0.05).argmin()
#         val_re.append(dfgrid2['Test_val'][ind_r])
#         prob_re.append(dfgrid2['Avg_prob_test'][ind_r])