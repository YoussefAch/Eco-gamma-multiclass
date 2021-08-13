from Stats_tests_utils import friedman_test
from TestUtils import buildBestOfGroupsName, buildExperimentName
import pickle
import os
import numpy as np

# scores_methods = []
# for method in methods:
#     meth_scores = []
#     for dataset in datasets:
#         meth_scores.append(df_metrics_opt[(df_metrics_opt['Dataset']==dataset) & (df_metrics_opt['timeParam']==dataset_tempcost[dataset]) & (df_metrics_opt['Method']==method)][metric].values[0])
#     scores_methods.append(meth_scores)

# iman_davenport, p_value, rankings_avg, rankings_cmp = friedman_test(r, *scores_methods)


# PATH_RESULTS_TORNADO = "experiments/results tornado/tor_2_exp2"
PATH_RESULTS_TORNADO = "experiments/results tornado/tor_2_exp8"
PATH_EXPERIMENT = PATH_RESULTS_TORNADO + "/experiment_tor_2"
EXPERIMENT_PARAMS_PATH = PATH_EXPERIMENT + "/inputParams.json"

# nbGroups = range(1, 20)
aggregates = ["entropy", "margins", "max", "gini"]
# methods = ["Gamma_MC_C1", "Gamma_MC", "K_MC"]

methods = ["Gamma_MC_C1", "Gamma_MC", "K_MC", "Gamma_MC_C1_Norm"]
methods_to_compare = [
    {
        "method": "Gamma_MC_C1",
        "nickname" : r"ECO-$\gamma$-Kmeans",
    },
    {
        "method": "Gamma_MC",
        "aggregate": "entropy",
        "nickname" : r"ECO-$\gamma$-entropy"
    },
    {
        "method": "Gamma_MC",
        "aggregate": "margins",
        "nickname" : r"ECO-$\gamma$-margins"
    },
    {
        "method": "Gamma_MC",
        "aggregate": "max",
        "nickname" : r"ECO-$\gamma$-max"
    },
    {
        "method": "Gamma_MC",
        "aggregate": "gini",
        "nickname" : r"ECO-$\gamma$-gini"
    },
    {
        "method": "K_MC",
        "nickname" : r"ECO-K"
    },
    {
        "method": "Gamma_MC_C1_Norm",
        "nickname" : r"ECO-$\gamma$-Kmeans-cal"
    },
]


# timeParams = [0.001, 0.01, 0.1, 0.5, 1]
timeParams = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]


# All datasets
PATH_FIGURES = PATH_RESULTS_TORNADO + "/figures"
datasets = ['Coffee', 'Beef', 'OliveOil', 'Lighting2', 'Lighting7', 'FaceFour', 'ECG200', 'Trace', 'Gun_Point', 'FISH', 'OSULeaf', 'Synthetic_control', 'DiatomSizeReduction', 'Haptics', 'Cricket_X', 'Cricket_Y', 'Cricket_Z', 'Adiac', '50words', 'InlineSkate', 'SonyAIBORobotSurface', 'SwedishLeaf', 'WordsSynonyms', 'MedicalImages', 'ECGFiveDays', 'CBF', 'SonyAIBORobotSurfaceII', 'Symbols', 'ItalyPowerDemand', 'TwoLeadECG', 'MoteStrain', 'CinC_ECG_torso', 'FaceAll', 'NonInvasiveFatalECG_Thorax1', 'NonInvasiveFatalECG_Thorax2', 'FacesUCR', 'MALLAT', 'Yoga', 'StarLightCurves', 'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y', 'uWaveGestureLibrary_Z', 'ChlorineConcentration', 'Two_Patterns', 'Wafer']

# Multiclass
# datasets = ["Beef", "OliveOil", "Lighting7", "FaceFour", "Trace", "FISH", "OSULeaf", "Synthetic_control", "DiatomSizeReduction", "Haptics", "Cricket_X", "Cricket_Y", "Cricket_Z", "Adiac", "50words", "InlineSkate", "SwedishLeaf", "WordsSynonyms", "MedicalImages",
#             "CBF", "Symbols", "CinC_ECG_torso", "FaceAll", "NonInvasiveFatalECG_Thorax1", "NonInvasiveFatalECG_Thorax2", "FacesUCR", "MALLAT", "StarLightCurves", "uWaveGestureLibrary_X", "uWaveGestureLibrary_Y", "uWaveGestureLibrary_Z", "ChlorineConcentration", "Two_Patterns"]
# PATH_FIGURES = PATH_RESULTS_TORNADO + "/figures_multiclass_only"
if not os.path.exists(PATH_FIGURES):
    os.mkdir(PATH_FIGURES)

# # Cleaned of strange datasets
# datasets = [ "FISH", "DiatomSizeReduction", "Cricket_X", "Cricket_Y", "Cricket_Z", "50words", "SwedishLeaf", "WordsSynonyms",
#             "CBF", "Symbols", "CinC_ECG_torso", "FaceAll", "NonInvasiveFatalECG_Thorax1", "FacesUCR", "MALLAT", "StarLightCurves", "uWaveGestureLibrary_X", "uWaveGestureLibrary_Y", "uWaveGestureLibrary_Z", "ChlorineConcentration", "Two_Patterns"]
# # PATH_FIGURES = PATH_RESULTS_TORNADO + "/figures_multiclass_only_cleaned"
# PATH_FIGURES = PATH_RESULTS_TORNADO + "/figures_multiclass_only"


if not os.path.exists(PATH_FIGURES):
    os.mkdir(PATH_FIGURES)


with open(os.path.join(PATH_EXPERIMENT, "intermediate_results", "bestGroup.pkl"), "rb") as f:
    best_groups_and_scores = pickle.load(f)

with open(os.path.join(PATH_EXPERIMENT, "results.pkl"), "rb") as g:
    results = pickle.load(g)

# methods_to_compare = []
# for method in methods:
#     if method == "Gamma_MC":
#         for aggregate in aggregates:
#             methods_to_compare.append(
#                 {"method": method, "aggregate": aggregate})
#     else:
#         methods_to_compare.append({"method": method})



orders_dict = {}
import Orange
import math
import matplotlib.pyplot as plt

for param_time in timeParams:
    scores = []

    for j, method in enumerate(methods_to_compare):
        method_scores = []
        for dataset in datasets:

            vp = {**method, "param_time": param_time, "dataset": dataset}
            best_group = best_groups_and_scores[buildBestOfGroupsName(vp)][0]
            method_scores.append(
                results[buildExperimentName({**vp, "n_groups": best_group})])        
        scores.append(method_scores)

    iman_davenport, p_value, rankings_avg, rankings_cmp = friedman_test(
        "petit", *scores)

    cd = Orange.evaluation.scoring.compute_CD(
        avranks=rankings_avg, n=len(datasets), alpha="0.05", test="nemenyi")

    rank_viz_fpath = PATH_FIGURES + "/Nemenyi_" + str(param_time) + ".png"
    lowv = math.floor(min(rankings_avg))
    highv = math.ceil(max(rankings_avg))
    width = (highv - lowv)*1.2 + 2 + 7

    orders_dict[param_time] = list(np.array(rankings_cmp).argsort()) 
    print( list(np.array(rankings_cmp).argsort()))

    Orange.evaluation.scoring.graph_ranks(
        filename=rank_viz_fpath,
        avranks=rankings_avg,
        names=[method["nickname"] for method in methods_to_compare],
        cd=cd,
        lowv=lowv,
        highv=highv,
        width=width,
        fontsize=3,
        textspace=5,
        reverse=False)    
    plt.close()


print(orders_dict)