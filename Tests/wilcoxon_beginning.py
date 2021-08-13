
# from Stats_tests_utils import matrix_ranking_vsSR, matrix_ranking, wilco_approche_adapt_vs_nonadapt, wilco_approches_vs_SR
# from utils import entropyFunc1, giniImpurity, sigmoid
import sys
from sklearn.metrics import cohen_kappa_score
from Stats_tests_utils import wilcoxon_test
import pickle
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from TestUtils import buildExperimentName, buildBestOfGroupsName, buildBestOfGroupsName, buildMethodName, buildModelName
import pandas as pd
"""
    1) Applies Wilcoxon test between each pair of techniques and displayss results
    2) For each time parameter, plots scores of datasets with best group for all methods
    3) Prints average scoers for each time param
"""


def concatenateDict(D):
    return(','.join(D.values()))


PATH_RESULTS_TORNADO = "experiments/results tornado/tor_2_exp8"
PATH_EXPERIMENT = PATH_RESULTS_TORNADO + "/experiment_tor_2"
EXPERIMENT_PARAMS_PATH = PATH_EXPERIMENT + "/inputParams.json"
PATH_REALDATA = "RealData-MORI"
PATH_PREDICTIONS = os.path.join(
    PATH_EXPERIMENT, "intermediate_results", "predictions")

SHOW = False

with open(os.path.join(PATH_EXPERIMENT, "intermediate_results", "bestGroup.pkl"), "rb") as f:
    best_groups_and_scores = pickle.load(f)
with open(os.path.join(PATH_EXPERIMENT, "results.pkl"), "rb") as g:
    results = pickle.load(g)
with open(os.path.join(PATH_EXPERIMENT, "test_scores_SR.json")) as h:
    test_scores_SR = json.load(h)

# for v in np.unique(np.array(list(best_groups_and_scores.keys()))):
#     print(v)

params = json.load(open(EXPERIMENT_PARAMS_PATH, "r"))
print(params)


# All datasets
# datasets = params["Datasets"]
# PATH_FIGURES = PATH_RESULTS_TORNADO + "/figures"

# Multiclass only
datasets = ["Beef", "OliveOil", "Lighting7", "FaceFour", "Trace", "FISH", "OSULeaf", "Synthetic_control", "DiatomSizeReduction", "Haptics", "Cricket_X", "Cricket_Y", "Cricket_Z", "Adiac", "50words", "InlineSkate", "SwedishLeaf", "WordsSynonyms", "MedicalImages", "CBF", "Symbols", "CinC_ECG_torso", "FaceAll", "NonInvasiveFatalECG_Thorax1", "NonInvasiveFatalECG_Thorax2", "FacesUCR", "MALLAT", "StarLightCurves", "uWaveGestureLibrary_X", "uWaveGestureLibrary_Y", "uWaveGestureLibrary_Z", "ChlorineConcentration", "Two_Patterns"]
PATH_FIGURES = PATH_RESULTS_TORNADO + "/figures_multiclass_only"

# Cleaned of strange datasets
# datasets = [ "FISH", "DiatomSizeReduction", "Cricket_X", "Cricket_Y", "Cricket_Z", "50words", "SwedishLeaf", "WordsSynonyms",
#             "CBF", "Symbols", "CinC_ECG_torso", "FaceAll", "NonInvasiveFatalECG_Thorax1", "FacesUCR", "MALLAT", "StarLightCurves", "uWaveGestureLibrary_X", "uWaveGestureLibrary_Y", "uWaveGestureLibrary_Z", "ChlorineConcentration", "Two_Patterns"]
# PATH_FIGURES = PATH_RESULTS_TORNADO + "/figures_multiclass_only_cleaned"

datasets_nb_classes = {'Coffee': 2, 'Beef': 5, 'OliveOil': 4, 'Lighting2': 2, 'Lighting7': 7, 'FaceFour': 4, 'ECG200': 2, 'Trace': 4, 'Gun_Point': 2, 'FISH': 7, 'OSULeaf': 6, 'Synthetic_control': 6, 'DiatomSizeReduction': 4, 'Haptics': 5, 'Cricket_X': 12, 'Cricket_Y': 12, 'Cricket_Z': 12, 'Adiac': 37, '50words': 50, 'InlineSkate': 7, 'SonyAIBORobotSurface': 2, 'SwedishLeaf': 15, 'WordsSynonyms': 25, 'MedicalImages': 10,
                       'ECGFiveDays': 2, 'CBF': 3, 'SonyAIBORobotSurfaceII': 2, 'Symbols': 6, 'ItalyPowerDemand': 2, 'TwoLeadECG': 2, 'MoteStrain': 2, 'CinC_ECG_torso': 4, 'FaceAll': 14, 'NonInvasiveFatalECG_Thorax1': 42, 'NonInvasiveFatalECG_Thorax2': 42, 'FacesUCR': 14, 'MALLAT': 8, 'Yoga': 2, 'StarLightCurves': 3, 'uWaveGestureLibrary_X': 8, 'uWaveGestureLibrary_Y': 8, 'uWaveGestureLibrary_Z': 8, 'ChlorineConcentration': 3, 'Two_Patterns': 4, 'Wafer': 2}
nbGroups = range(1, params["nbGroups"])
timeParams = params["timeParams"]
aggregates = ["entropy", "margins", "max", "gini"]
methods = ["Gamma_MC_C1", "Gamma_MC", "K_MC", "Gamma_MC_C1_Norm"]


# All methods we cant to compare with eachother
# methods_to_compare = []
# for method in methods:
#     if method == "Gamma_MC":
#         for aggregate in aggregates:
#             methods_to_compare.append(
#                 {"method": method, "aggregate": aggregate})
#     else:
#         methods_to_compare.append({"method": method})

####################################################################################################
# exp2
# orders_nemenyi = {0.001: [5, 1, 4, 3, 2, 0], 0.01: [4, 5, 1, 3, 2, 0], 0.1: [3, 4, 1, 0, 2, 5], 0.2: [3, 4, 1, 2, 0, 5], 0.3: [4, 3, 5, 1, 0, 2], 0.4: [4, 3, 2, 5, 0, 1], 0.5: [4, 5, 3, 0, 2, 1], 0.6: [4, 5, 0, 1, 3, 2], 0.7: [4, 3, 1, 5, 0, 2], 0.8: [4, 5, 3, 2, 0, 1], 0.9: [3, 5, 4, 0, 1, 2], 1: [5, 3, 1, 0, 2, 4]}

# # order exp4
# orders_nemenyi =  {0.001: [5, 4, 3, 2, 1, 0], 0.01: [5, 4, 3, 1, 2, 0], 0.1: [3, 1, 4, 0, 5, 2], 0.2: [3, 1, 4, 5, 0, 2], 0.3: [3, 4, 5, 1, 0, 2], 0.4: [3, 4, 2, 5, 1, 0], 0.5: [3, 4, 5, 1, 2, 0], 0.6: [5, 3, 4, 0, 2, 1], 0.7: [4, 5, 3, 1, 0, 2], 0.8: [4, 5, 3, 1, 2, 0], 0.9: [5, 3, 0, 4, 2, 1], 1: [5, 3, 1, 2, 0, 4]}
# # order exp4_all_datasets
# orders_nemenyi = {0.001: [5, 4, 3, 2, 1, 0], 0.01: [5, 4, 3, 1, 2, 0], 0.1: [3, 1, 4, 2, 5, 0], 0.2: [3, 1, 4, 5, 2, 0], 0.3: [4, 3, 5, 1, 2, 0], 0.4: [4, 3, 2, 1, 5, 0], 0.5: [4, 3, 5, 1, 2, 0], 0.6: [5, 4, 3, 2, 1, 0], 0.7: [3, 4, 5, 1, 2, 0], 0.8: [4, 5, 3, 1, 2, 0], 0.9: [3, 5, 4, 1, 2, 0], 1: [1, 3, 5, 4, 2, 0]}

# order exp5_alldatasets
# orders_nemenyi = {0.001: [5, 4, 3, 2, 1, 0], 0.01: [5, 4, 3, 1, 2, 0], 0.1: [3, 1, 4, 2, 5, 0], 0.2: [3, 1, 4, 5, 2, 0], 0.3: [4, 3, 5, 1, 2, 0], 0.4: [4, 3, 2, 1, 5, 0], 0.5: [4, 3, 5, 1, 2, 0], 0.6: [5, 4, 3, 1, 2, 0], 0.7: [3, 4, 5, 1, 2, 0], 0.8: [5, 4, 3, 1, 2, 0], 0.9: [5, 3, 1, 4, 2, 0], 1: [1, 3, 5, 4, 2, 0]}
# order exp5 MULTICLASS ONLY
# orders_nemenyi = {0.001: [5, 4, 3, 1, 2, 0], 0.01: [5, 4, 3, 1, 2, 0], 0.1: [3, 1, 4, 0, 5, 2], 0.2: [3, 1, 4, 5, 0, 2], 0.3: [3, 4, 5, 1, 0, 2], 0.4: [3, 4, 5, 2, 1, 0], 0.5: [3, 4, 5, 1, 2, 0], 0.6: [5, 4, 3, 0, 2, 1], 0.7: [4, 3, 5, 1, 0, 2], 0.8: [5, 4, 3, 1, 2, 0], 0.9: [5, 3, 0, 1, 4, 2], 1: [5, 3, 1, 2, 0, 4]}

# order exp6 MULTICLASS ONLY
# orders_nemenyi = {0.001: [5, 1, 4, 3, 2, 0], 0.01: [5, 1, 4, 3, 2, 0], 0.1: [3, 4, 1, 5, 2, 0], 0.2: [3, 1, 4, 2, 0, 5], 0.3: [4, 3, 5, 1, 0, 2], 0.4: [4, 3, 2, 5, 1, 0], 0.5: [5, 4, 3, 0, 2, 1], 0.6: [4, 5, 3, 1, 0, 2], 0.7: [4, 3, 5, 1, 0, 2], 0.8: [5, 4, 3, 2, 0, 1], 0.9: [5, 4, 3, 0, 1, 2], 1: [5, 1, 3, 2, 4, 0]}

# order exp7 MULTICLASS ONLY
# orders_nemenyi ={0.001: [5, 1, 4, 3, 2, 6, 0], 0.01: [5, 4, 1, 3, 2, 6, 0], 0.1: [3, 4, 1, 6, 0, 2, 5], 0.2: [3, 4, 1, 6, 0, 2, 5], 0.3: [4, 3, 5, 1, 0, 6, 2], 0.4: [4, 3, 2, 5, 0, 1, 6], 0.5: [5, 4, 3, 0, 1, 2, 6], 0.6: [4, 5, 1, 3, 0, 2, 6], 0.7: [4, 3, 1, 5, 0, 2, 6], 0.8: [4, 3, 2, 5, 1, 0, 6], 0.9: [3, 5, 4, 0, 1, 2, 6], 1: [3, 5, 1, 0, 2, 4, 6]}

# order exp8 MULTICLASS ONLY
# orders_nemenyi = {0.001: [5, 4, 1, 3, 2, 6, 0], 0.01: [5, 1, 4, 3, 2, 6, 0], 0.1: [3, 4, 1, 6, 5, 0, 2], 0.2: [3, 1, 4, 6, 2, 0, 5], 0.3: [4, 3, 5, 1, 6, 0, 2], 0.4: [
#     4, 3, 2, 5, 1, 0, 6], 0.5: [5, 4, 3, 0, 1, 2, 6], 0.6: [4, 5, 3, 1, 0, 2, 6], 0.7: [4, 3, 5, 1, 0, 6, 2], 0.8: [5, 4, 3, 2, 0, 1, 6], 0.9: [5, 3, 4, 0, 1, 2, 6], 1: [1, 5, 3, 2, 4, 0, 6]}

# order exp8 ALL DATASETS
orders_nemenyi = {0.001: [5, 4, 1, 2, 3, 6, 0], 0.01: [5, 1, 4, 3, 2, 6, 0], 0.1: [3, 4, 1, 6, 2, 5, 0], 0.2: [3, 1, 4, 2, 6, 0, 5], 0.3: [4, 3, 1, 5, 6, 2, 0], 0.4: [
    4, 3, 2, 1, 5, 6, 0], 0.5: [4, 5, 3, 1, 2, 6, 0], 0.6: [4, 5, 1, 3, 2, 0, 6], 0.7: [4, 3, 5, 1, 2, 0, 6], 0.8: [4, 5, 3, 2, 1, 6, 0], 0.9: [3, 4, 5, 1, 2, 0, 6], 1: [3, 1, 5, 4, 2, 0, 6]}
####################################################################################################


# Cleaned
# orders_nemenyi_OLD = {0.001: [5, 2, 1, 4, 3, 0], 0.01: [5, 1, 2, 4, 3, 0], 0.1: [3, 1, 4, 2, 0, 5], 0.2: [3, 1, 4, 2, 0, 5], 0.3: [3, 4, 1, 5, 2, 0], 0.4: [3, 4, 1, 2, 5, 0], 0.5: [4, 3, 1, 5, 0, 2], 0.6: [4, 3, 1, 5, 2, 0], 0.7: [1, 4, 3, 2, 5, 0], 0.8: [1, 4, 3, 2, 5, 0], 0.9: [4, 3, 1, 0, 5, 2], 1: [3, 1, 4, 0, 5, 2]}

# order the methods on their ranking

methods_to_compare = [

    {
        "method": "Gamma_MC",
        "aggregate": "entropy",
        "nickname": r"ECO-$\gamma$-entropy"
    },
    {
        "method": "Gamma_MC",
        "aggregate": "gini",
        "nickname": r"ECO-$\gamma$-gini"
    },
    {
        "method": "Gamma_MC",
        "aggregate": "margins",
        "nickname": r"ECO-$\gamma$-margins"
    },
    {
        "method": "Gamma_MC",
        "aggregate": "max",
        "nickname": r"ECO-$\gamma$-max"
    },
    {
        "method": "Gamma_MC_C1",
        "nickname": r"ECO-$\gamma$-Kmeans",
    },
    {
        "method": "Gamma_MC_C1_Norm",
        "nickname": r"ECO-$\gamma$-Kmeans-cal"
    },
    {
        "method": "K_MC",
        "nickname": r"ECO-K"
    },
]


# # ############################################################################################
# # Wilcoxon tests and plots
# # ############################################################################################

# PATH_WILCOXON = os.path.join(PATH_FIGURES, "Wilcoxon")
# if not os.path.exists(PATH_WILCOXON):
#     os.mkdir(PATH_WILCOXON)

# for param_time in timeParams:
#     methods_to_compare_matrix = [methods_to_compare[k]
#                                  for k in orders_nemenyi[param_time]]
#     print(methods_to_compare)
#     plt.figure()
#     pairewise_comparison_df = pd.DataFrame(
#         np.zeros(shape=(len(methods_to_compare_matrix),
#                         len(methods_to_compare_matrix))),
#         columns=[method["nickname"] for method in methods_to_compare_matrix],
#         index=[method["nickname"] for method in methods_to_compare_matrix],
#     )

#     for i, method_A in enumerate(methods_to_compare_matrix):
#         for j, method_B in enumerate(methods_to_compare_matrix):
#             A = []
#             B = []
#             for dataset in datasets:
#                 vp_A = {**method_A, "param_time": param_time, "dataset": dataset}
#                 vp_B = {**method_B, "param_time": param_time, "dataset": dataset}
#                 # A.append(best_groups_and_scores[buildBestOfGroupsName(vp_A)][1])
#                 # B.append(best_groups_and_scores[buildBestOfGroupsName(vp_B)][1])

#                 best_group_A = best_groups_and_scores[buildBestOfGroupsName(
#                     vp_A)][0]
#                 best_group_B = best_groups_and_scores[buildBestOfGroupsName(
#                     vp_B)][0]

#                 A.append(results[buildExperimentName(
#                     {**vp_A, "n_groups": best_group_A})])
#                 B.append(results[buildExperimentName(
#                     {**vp_B, "n_groups": best_group_B})])

#             # Pretty print and plot
#             z, reject = wilcoxon_test(A, B)
#             temp = " ".join(("Alpha =", str(
#                 param_time) + ",", concatenateDict(method_A), "vs", concatenateDict(method_B)))
#             temp += "-"*(60-len(temp)) + " " + str((z, reject))
#             temp += "-"*(100-len(temp))
#             if z < 0:
#                 temp += "PERD"
#             else:
#                 temp += "GAGNE"

#             # Plot matrix
#             if (reject):
#                 #     temp += "-"*(125-len(temp)) + "SIGNIF."
#                 #     if z< 0:
#                 #         pairewise_comparison_df[concatenateDict(method_A)][concatenateDict(method_B)] = -0.7
#                 #     elif z > 0:
#                 #         pairewise_comparison_df[concatenateDict(method_A)][concatenateDict(method_B)] = +0.7
#                 pairewise_comparison_df[method_A["nickname"]
#                                         ][method_B["nickname"]] = 1

#             else:
#                 #     if z< 0:
#                 #         pairewise_comparison_df[concatenateDict(method_A)][concatenateDict(method_B)] = -0.2
#                 #     elif z > 0:
#                 #         pairewise_comparison_df[concatenateDict(method_A)][concatenateDict(method_B)] = +0.2
#                 pairewise_comparison_df[method_A["nickname"]
#                                         ][method_B["nickname"]] = 0

#             # plt.text(i, j, round(z, 2), ha='center', va='center')
#             print(temp)
#         print()

#     plt.xticks(np.arange(0, len(pairewise_comparison_df)),
#                pairewise_comparison_df.columns, rotation='vertical', fontsize=50)
#     plt.yticks(np.arange(0, len(pairewise_comparison_df)),
#                pairewise_comparison_df.columns, fontsize=50)
#     plt.tick_params(axis='y', which='both', labelleft=False, labelright=True)
#     # plt.title("Wilcoxon tests, alpha = " + str(param_time))
#     plt.gcf().set_size_inches(9, 9)

#     # Color
#     # plt.imshow(pairewise_comparison_df, cmap=plt.cm.RdYlGn, vmin=-1.0, vmax=1.0)
#     # Black and white
#     plt.imshow(pairewise_comparison_df, cmap=plt.cm.gray, vmin=0, vmax=1.0)

#     plt.savefig(os.path.join(PATH_WILCOXON, "alpha=" +
#                              str(param_time)+".png"), bbox_inches='tight')

#     if SHOW:
#         plt.show()
#     print()
#     print()
#     print()


##############################################################################################
# # PLOT SCORES ON ALL DATASETS FOR EACH TIME PARAM
##############################################################################################

# for time_param in timeParams:
#     X = []
#     Y_methods = [[] for method in methods_to_compare]

#     plt.figure()
#     for dataset in datasets:
#         X.append(dataset)
#         for i,method in enumerate(methods_to_compare):
#             Y_methods[i].append(best_groups_and_scores[buildBestOfGroupsName({**method,"param_time": time_param, "dataset": dataset})][1])

#     for i,method in enumerate(methods_to_compare):
#         plt.scatter(X, Y_methods[i], label= method["nickname"], s= 100, alpha=0.4)

#     plt.title("alpha = " + str(time_param))
#     plt.legend()
#     plt.xticks(rotation=90)
#     plt.gcf().set_size_inches(20, 10)
#     plt.savefig(os.path.join(PATH_FIGURES, "scores_detailed" +str(time_param) + ".png"))
#     if SHOW:
#         plt.show()


# # Prints average scores for each method and time parameter
# for param_time in timeParams:
#     print("Alpha =", param_time)
#     for method in methods_to_compare:
#         scores = [best_groups_and_scores[buildBestOfGroupsName({**method,"param_time": param_time, "dataset": dataset})][1] for dataset in datasets]
#         print(method["nickname"], ":", np.mean(np.array(scores)))
#     print()


# # Prints score and number of groups corresponding
# for param_time in timeParams:
#     print("Alpha =", param_time)
#     for dataset in datasets:
#         print("\t", dataset)
#         for method in methods_to_compare:
#             s  = best_groups_and_scores[buildBestOfGroupsName({**method,"param_time": param_time, "dataset": dataset})]
#             temp = method["nickname"]
#             print("\t\t" + temp , "-"*(40-len(temp)), s)
#     print()


############################################################################################
# Comparative Wilcoxon plot (grids)
############################################################################################

markers = ['o', 'o', 'o', 'o']

PATH_WILCO_GRID = os.path.join(PATH_FIGURES, "wilco_grid")

if not os.path.exists(PATH_WILCO_GRID):
    os.mkdir(PATH_WILCO_GRID)

methods_opponent = methods_to_compare + [{"method": "SR", "nickname": "SR"}]

for i, method_A in enumerate(methods_opponent):
    plt.figure(figsize=(7, 2))
    if method_A["method"] == "SR":
        methods_to_compare_comparative = methods_to_compare
    else:
        methods_to_compare_comparative = [
            method for method in methods_to_compare if method["nickname"] != method_A["nickname"]]

    print("AH")

    ticks_y = []

    for j, method_B in enumerate(methods_to_compare_comparative):
        wilco_results = []

        for param_time in timeParams:
            A = []
            B = []
            for dataset in datasets:

                if method_A["method"] == "SR":
                    # SR
                    A.append(test_scores_SR[str(param_time) + ',' + dataset])
                else:
                    vp_A = {**method_A, "param_time": param_time,
                            "dataset": dataset}
                    best_group_A = best_groups_and_scores[buildBestOfGroupsName(
                        vp_A)][0]
                    A.append(results[buildExperimentName(
                        {**vp_A, "n_groups": best_group_A})])

                vp_B = {**method_B, "param_time": param_time, "dataset": dataset}
                best_group_B = best_groups_and_scores[buildBestOfGroupsName(
                    vp_B)][0]
                B.append(results[buildExperimentName(
                    {**vp_B, "n_groups": best_group_B})])

            z, reject = wilcoxon_test(A, B)
            wilco_results.append((z, reject))

        significant_won = [k for k, (z, reject) in enumerate(
            wilco_results) if reject and z < 0]
        significant_lost = [k for k, (z, reject) in enumerate(
            wilco_results) if reject and z > 0]
        not_significant = [k for k, (z, reject) in enumerate(
            wilco_results) if not reject]

        y_value = len(methods_to_compare_comparative) - j
        ticks_y.append(y_value)

        # y_value = 3
        plt.scatter(significant_won, [y_value] *
                    len(significant_won), marker='+', color='black')
        plt.scatter(significant_lost, [
                    y_value]*len(significant_lost), marker='>', color='black')
        plt.scatter(not_significant, [
                    y_value]*len(not_significant), color='black', facecolor='white')

    plt.xlabel(r'$\alpha$', fontsize=18)
    # y_axis = np.arange(1, len(methods_to_compare_comparative) + 1)
    print([method["nickname"] for method in methods_to_compare_comparative])
    plt.yticks(ticks_y, [method["nickname"]
                        for method in methods_to_compare_comparative])
    # plt.xticks(timeParams, rotation=90)
    plt.xticks([k for k in range(len(timeParams))], timeParams, rotation=90)
    # print(y_axis) 

    plt.savefig(os.path.join(PATH_WILCO_GRID, "grid_" +
                             buildMethodName(method_A) + ".png"), bbox_inches='tight')

    if SHOW:
        plt.show()
    plt.close()


# #############################################################################################
# #  KAPPA AND FRONT
# #############################################################################################


# def buildExpNameFup(p):
#     name = vp["method"] + ',' + vp["dataset"] + ',' + str(vp["n_groups"])
#     name += ',' + str(param_time)

#     if vp["method"] == "Gamma_MC":
#         name += ',' + vp["aggregate"]
#     return name


# # method = {"method" : "K_MC"}
# # param_time = 0.1
# markers = ["o", "s", "^", "*"]
# fill_styles = ["full", "none"]
# markercolors = [None, "white"]
# # nickname_dashed = [r"ECO-$\gamma$-C1", r"ECO-$\gamma$-marg", r"K-MC", r"ECO-$\gamma$-Kmeans"]
# nickname_dashed = ['ECO-$\\gamma$-Kmeans',
#                    'ECO-$\\gamma$-margins', 'ECO-K', 'ECO-$\\gamma$-Kmeans-cal']
# save_dict = {}

# computeAll = True

# plt.figure(figsize=(10, 8))
# # plt.figure(figsize = (12,10))

# for i, method in enumerate(methods_to_compare):

#     X = []
#     Y = []
#     print(method["nickname"])

#     if computeAll:
#         for param_time in timeParams:
#             print(param_time)
#             kappas = []
#             taus = []
#             for dataset in datasets:
#                 filepathTest = PATH_REALDATA + '/'+dataset+'/' + dataset+'_TEST_SCORE.tsv'
#                 test = pd.read_csv(filepathTest, sep='\t',
#                                    header=None, index_col=None, engine='python')
#                 y_test = test.iloc[:, 0]

#                 X_test = test.loc[:, test.columns != test.columns[0]]
#                 max_t = X_test.shape[1]

#                 vp = {**method, "dataset": dataset, "param_time": param_time}
#                 vp["n_groups"] = best_groups_and_scores[buildBestOfGroupsName(
#                     vp)][0]

#                 with open(os.path.join(PATH_PREDICTIONS, "PREDECO" + buildExpNameFup(vp) + ".pkl"), "rb") as file_pred:
#                     predeco = np.array(
#                         list(pickle.load(file_pred).values())[0])

#                 kappa = cohen_kappa_score(y_test, list(predeco[:, 2]))
#                 kappas.append(kappa)
#                 taus.append(np.median(predeco[:, 0])/max_t)

#             # print(np.mean(kappas))
#             # print(np.median(kappas))
#             # print(np.mean(taus))
#             # print(np.median(taus))

#             X.append(np.mean(taus))
#             # X.append(np.median(taus))
#             Y.append(np.mean(kappas))
#             # Y.append(np.median(kappas))

#             save_dict[method["nickname"]] = {"taus": X, "kappas": Y}
#             print()
#     else:
#         kappas_taus = json.load(
#             open(os.path.join(PATH_EXPERIMENT, "kappas_median.json"), "r"))
#         X = kappas_taus[method["nickname"]]["taus"]
#         Y = kappas_taus[method["nickname"]]["kappas"]

#     # Plot
#     if method["nickname"] in nickname_dashed:
#         dash = '--'
#     else:
#         dash = '-'

#     plt.plot(X, Y, dash, label=method["nickname"], marker=markers[i//2],
#              markerfacecolor=markercolors[i % 2], fillstyle=fill_styles[i % 2])
#     alp = r'$\alpha$ = '
#     if method["method"] == "Gamma_MC_C1":
#         for i in [0, 2, 4, 6, 8, 9, 10]:
#             plt.annotate(alp + str(timeParams[i]), (X[i]+0.005, Y[i]-0.01))

#     plt.xlabel(r'$Earliness$', fontsize=16)
#     plt.ylabel(r'$Kappa$', fontsize=16)
#     print()
#     print()
#     print()

# if computeAll:
#     json.dump(save_dict, open(os.path.join(
#         PATH_EXPERIMENT, "kappas_median.json"), "w"))

# plt.legend()
# plt.savefig(os.path.join(PATH_FIGURES, "front.png"))
# # plt.show()


##############################################################################################
# Histogram nb classes
##############################################################################################

# plt.hist(list(datasets_nb_classes.values()))
# plt.title("Repartition of number of classes (Mori datasets)")
# plt.xlabel("Number of classes")
# plt.ylabel("Counts")
# plt.show()


##############################################################################################
# Plot matrices
##############################################################################################
# for dataset in datasets:
#     for n_groups in nbGroups:
#         vp = {"method": "Gamma_MC_C1", "dataset":dataset, "n_groups": n_groups}

#         with open(os.path.join("experiments/results tornado/tor_2_exp1/experiment_tor_2", "modelsECONOMY", buildModelName(vp)+ ".pkl" ), 'rb') as f:
#             model  = pickle.load(f)
#         for t in model.transitionMatrices.keys():
#             plt.close()
#             plt.matshow(model.transitionMatrices[t], cmap = 'gray')
#             plt.title(dataset + " - " + "K = " + str(n_groups) + " - t = " + str(t) )
#             print(model.transitionMatrices[t])
#             plt.savefig("matrices/" + dataset + "_" + str(n_groups) + "_" + str(t) + ".png")
#             # plt.show()

##############################################################################################
# Compare GINI / Entropy confidence scores
##############################################################################################

sys.path.append("..")


# datasets = ["Cricket_X"]


# errors_dict = {key: 0 for key in datasets}
# for dataset in datasets:
#     print(dataset)
#     vp = {"method": "Gamma_MC_C1", "dataset":dataset, "n_groups": 1}
#     with open(os.path.join("experiments/results tornado/tor_2_exp1/experiment_tor_2", "modelsECONOMY", buildModelName(vp)+ ".pkl" ), 'rb') as f:
#         model  = pickle.load(f)

#     for t in model.transitionMatrices.keys():

#         with open(os.path.join(PATH_REALDATA, dataset, "ep_probas_" + str(t)+ ".pkl"), 'rb') as g:
#             probas = pickle.load(g)

#         entropy = [sigmoid(entropyFunc1(x)) for x in probas]
#         gini = [sigmoid(giniImpurity(x)) for x in probas]

#         # entropy = np.array([entropyFunc1(x) for x in probas])
#         # gini = np.array([giniImpurity(x) for x in probas])

#         sorted_arg_entropy = np.argsort(entropy)
#         sorted_arg_gini = np.argsort(gini)

#         equals = np.array_equal(sorted_arg_entropy, sorted_arg_gini)
#         print(t,equals )

#         if (not equals):
#             errors_dict[dataset] += 1

# print()
# print()
# print()
# print(errors_dict)
