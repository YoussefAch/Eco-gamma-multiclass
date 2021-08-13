from Stats_tests_utils import wilcoxon_test
import pickle
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from TestUtils import buildExperimentName, buildBestOfGroupsName, buildBestOfGroupsName, buildMethodName, buildModelName
import pandas as pd

PATH_RESULTS_TORNADO = "experiments/results tornado"
# PATH_EXPERIMENT = PATH_RESULTS_TORNADO + "/experiment_tor_2"
# EXPERIMENT_PARAMS_PATH = PATH_EXPERIMENT + "/inputParams.json"
PATH_REALDATA = "RealData-MORI"
# PATH_PREDICTIONS = os.path.join(PATH_EXPERIMENT, "intermediate_results", "predictions")

SHOW = False

# Multiclass only
datasets = ["Beef", "OliveOil", "Lighting7", "FaceFour", "Trace", "FISH", "OSULeaf", "Synthetic_control", "DiatomSizeReduction", "Haptics", "Cricket_X", "Cricket_Y", "Cricket_Z", "Adiac", "50words", "InlineSkate", "SwedishLeaf", "WordsSynonyms", "MedicalImages", "CBF", "Symbols", "CinC_ECG_torso", "FaceAll", "NonInvasiveFatalECG_Thorax1", "NonInvasiveFatalECG_Thorax2", "FacesUCR", "MALLAT", "StarLightCurves", "uWaveGestureLibrary_X", "uWaveGestureLibrary_Y", "uWaveGestureLibrary_Z", "ChlorineConcentration", "Two_Patterns"]
PATH_FIGURES = PATH_RESULTS_TORNADO + "/figures_multiclass_only"

exps = ["exp2", "exp4", "exp5", "exp6"]
best_groups_and_scores_dict = {}

methods_to_compare = [
    {
        "method": "Gamma_MC_C1",
        "nickname" : r"$\gamma$MC-C1"
    },
    {
        "method": "Gamma_MC",
        "aggregate": "entropy",
        "nickname" : r"$\gamma$MC-entpy"
    },
    {
        "method": "Gamma_MC",
        "aggregate": "margins",
        "nickname" : r"$\gamma$MC-marg"
    },
    {
        "method": "Gamma_MC",
        "aggregate": "max",
        "nickname" : r"$\gamma$MC-max"
    },
    {
        "method": "Gamma_MC",
        "aggregate": "gini",
        "nickname" : r"$\gamma$MC-gini"
    },
    {
        "method": "K_MC",
        "nickname" : r"K-MC"
    },
]

timeParams = [0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

for exp in exps:
    with open(os.path.join(PATH_RESULTS_TORNADO, "tor_2_"+exp, "experiment_tor_2", "intermediate_results", "bestGroup.pkl"), "rb") as f:
        best_groups_and_scores = pickle.load(f)
    best_groups_and_scores_dict[exp] = best_groups_and_scores

# Prints score and number of groups corresponding
for method in methods_to_compare:
    print(buildMethodName(method))
    for param_time in timeParams:
        print("Alpha =", param_time)
        nb_groups = {dataset: [best_groups_and_scores_dict[exp][buildBestOfGroupsName({**method,"param_time": param_time, "dataset": dataset})][0] for exp in exps] for dataset in datasets}
        # print(dataset, nb_groups)
        df = pd.DataFrame.from_dict(nb_groups, orient = 'index')
        df.to_csv(buildMethodName(method)+','+str(param_time)+".csv")

