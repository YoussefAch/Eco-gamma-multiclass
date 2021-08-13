import sys
import os.path
import argparse
import json
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import multiprocessing
try:
    import cPickle as pickle
except ImportError:
    import pickle


def scoreSR(arguments):

    SRParams, timecostParam, ep_probas, ep_preds, val_probas, val_preds, y_test_ep, y_test_val, nb_observations_val, nb_observations_ep, timestamps, timecost, max_t = arguments
    score = 0

    for i in range(nb_observations_ep):
        for t in timestamps:
            maxiProba, scndProba = get_pair_max_elements(ep_probas[t][i])

            # Stopping rule
            sr = SRParams[0] * maxiProba + SRParams[1] * \
                (maxiProba-scndProba) + SRParams[2] * (t / max_t)

            if sr > 0 or t == timestamps[-1]:
                if y_test_ep.iloc[i] == ep_preds[t][i]:
                    score += timecost[t]
                else:
                    score += timecost[t] + 1  # C_m = 1
                break

    for i in range(nb_observations_val):
        for t in timestamps:

            maxiProba, scndProba = get_pair_max_elements(val_probas[t][i])

            # Stopping rule
            sr = SRParams[0] * maxiProba + SRParams[1] * \
                (maxiProba-scndProba) + SRParams[2] * (t / max_t)

            if sr > 0 or t == timestamps[-1]:
                if y_test_val.iloc[i] == val_preds[t][i]:
                    score += timecost[t]
                else:
                    score += timecost[t] + 1  # C_m = 1
                break

    # print('------------FINISH---------- : ', timecostParam)
    return score / (nb_observations_ep + nb_observations_val)


def predictSR(arguments):
    print('DEBUT !! ')

    SRParams, sampling_ratio, folderRealData, dataset, timecostParam = arguments

    with open(folderRealData+'/'+dataset+'/test_probas.pkl', 'rb') as inp:
        test_probas = pickle.load(inp)
    with open(folderRealData+'/'+dataset+'/test_preds.pkl', 'rb') as inp:
        test_preds = pickle.load(inp)

    # compute AvgCost
    filepathTest = folderRealData+'/'+dataset+'/'+dataset+'_TEST_SCORE.tsv'
    test = pd.read_csv(filepathTest, sep='\t', header=None,
                       index_col=None, engine='python')
    li, col = test.shape
    max_t = col - 1
    step = int(max_t*sampling_ratio) if int(max_t*sampling_ratio) > 0 else 1
    timestamps = [t for t in range(step, max_t + 1, step)]
    mn = np.min(np.unique(test.iloc[:, 0].values))
    timecost = timecostParam * np.arange(max_t+1)



    
    test.iloc[:, 0] = test.iloc[:, 0].apply(lambda e: 0 if e == mn else 1)
    # get X and y
    y_test = test.iloc[:, 0]
    del test
    nb_observations = li

    # compute trigger times
    trigger_times = []
    score = 0
    preds = []
    for i in range(nb_observations):
        for t in timestamps:
            maxiProba, scndProba = get_pair_max_elements(test_probas[t][i])
            # Stopping rule
            sr = SRParams[0] * maxiProba + SRParams[1] * \
                (maxiProba-scndProba) + SRParams[2] * (t / max_t)

            if sr > 0 or t == timestamps[-1]:
                preds.append(test_preds[t][i])
                trigger_times.append(t)
                if y_test.iloc[i] == test_preds[t][i]:
                    score += timecost[t]
                else:
                    score += timecost[t] + 1  # C_m = 1
                break
    score = score/nb_observations
    print('score : ', score)

    # kapa = round(cohen_kappa_score(y_test.values.tolist(), preds), 2)
    # print('kapa : ', kapa)
    # colms = ['Dataset', 'timeParam', 'Score', 'Med_tau', 'Kappa']
    # v = [dataset, timecostParam, score,
    #      np.median(np.array(trigger_times)), kapa]

    # return {key: value for key, value in zip(colms, v)}


    return {
            "dataset": dataset,
            "param_time": timecostParam,
            "SRparams": SRParams,
            "test_score": score
        }


def get_pair_max_elements(vector):
    sorted = np.sort(vector)
    return (sorted[-1], sorted[-2])


# filename = "experiments/experiment_SR_approach/df_metrics_SR.pkl"
# with open(filename, 'rb') as input:
#     df_metrics_opt = pickle.load(input)

# print(df_metrics_opt)
# # methods = np.unique(df_metrics_opt['Method'])
# datasets = list(np.unique(df_metrics_opt['Dataset']))
# # datasets = list(np.unique(df_metrics_opt['Dataset']))[:2]
# timeParams = list(np.unique(df_metrics_opt['timeParam']))
# timeParams = list(map(float, timeParams))
# timeParams.sort()
# timeParams = list(map(str, timeParams))
# timeParams.pop()
# timeParams.append('1')
datasets = ['Coffee', 'Beef', 'OliveOil', 'Lighting2', 'Lighting7', 'FaceFour', 'ECG200', 'Trace', 'Gun_Point', 'FISH', 'OSULeaf', 'Synthetic_control', 'DiatomSizeReduction', 'Haptics', 'Cricket_X', 'Cricket_Y', 'Cricket_Z', 'Adiac', '50words', 'InlineSkate', 'SonyAIBORobotSurface', 'SwedishLeaf', 'WordsSynonyms', 'MedicalImages', 'ECGFiveDays', 'CBF',
            'SonyAIBORobotSurfaceII', 'Symbols', 'ItalyPowerDemand', 'TwoLeadECG', 'MoteStrain', 'CinC_ECG_torso', 'FaceAll', 'NonInvasiveFatalECG_Thorax1', 'NonInvasiveFatalECG_Thorax2', 'FacesUCR', 'MALLAT', 'Yoga', 'StarLightCurves', 'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y', 'uWaveGestureLibrary_Z', 'ChlorineConcentration', 'Two_Patterns', 'Wafer']
timeParams = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# timeParams = [0.001]


folderRealData = 'RealData-MORI'
nb_core = multiprocessing.cpu_count()
nb_core = 4
sampling_ratio = 0.05


# # SR params to determine
# SRParamsOptimized = {timeparam: {dataset: []}
#                      for timeparam in timeParams for dataset in datasets}
# # candidates_parameters = np.linspace(-1, 1, 41)


# candidates_parameters = np.linspace(-1, 1, 21)
# possibles_gammas = [[g1, g2, g3]
#                     for g1 in candidates_parameters for g2 in candidates_parameters for g3 in candidates_parameters]

# for dataset in datasets:
#     print("Running for: ", dataset)
#     with open(folderRealData+'/'+dataset+'/ep_probas.pkl', 'rb') as inp:
#         ep_probas = pickle.load(inp)
#     with open(folderRealData+'/'+dataset+'/ep_preds.pkl', 'rb') as inp:
#         ep_preds = pickle.load(inp)

#     with open(folderRealData+'/'+dataset+'/val_probas.pkl', 'rb') as inp:
#         val_probas = pickle.load(inp)
#     with open(folderRealData+'/'+dataset+'/val_preds.pkl', 'rb') as inp:
#         val_preds = pickle.load(inp)

#     filepathep = folderRealData+'/'+dataset+'/'+dataset+'_ESTIMATE_PROBAS.tsv'
#     ep = pd.read_csv(filepathep, sep='\t', header=None,
#                      index_col=None, engine='python')
#     nb_observations_ep, col = ep.shape
#     max_t = col-1
#     step = int(max_t*sampling_ratio) if int(max_t*sampling_ratio) > 0 else 1
#     timestamps = [t for t in range(step, max_t + 1, step)]

#     # get X and y
#     y_test_ep = ep.iloc[:, 0]

#     filepathval = folderRealData+'/'+dataset+'/'+dataset+'_VAL_SCORE.tsv'
#     val = pd.read_csv(filepathval, sep='\t', header=None,
#                       index_col=None, engine='python')
#     nb_observations_val, col = val.shape
#     y_test_val = val.iloc[:, 0]

#     del ep, val

#     for timecostParam in timeParams:
#         timecost = float(timecostParam) * np.arange(max_t+1)
#         args_parallel = [[elm, timecostParam, ep_probas, ep_preds, val_probas, val_preds, y_test_ep, y_test_val,
#                           nb_observations_val, nb_observations_ep, timestamps, timecost, max_t] for elm in possibles_gammas]
#         predictions = Parallel(n_jobs=nb_core)(
#             delayed(scoreSR)(func_arg) for func_arg in args_parallel)

#         index = np.argmin(np.array(predictions))

#         print(predictions[index])
#         print('time : ', timecostParam, 'Dataset',
#               dataset, ' ', possibles_gammas[index])
#         SRParamsOptimized[timecostParam][dataset] = possibles_gammas[index]

#         with open('experiments/experiment_SR_approach/' + str(timecostParam) + ',' + dataset + '.pkl', 'wb') as inp:
#             pickle.dump(
#                 {
#                     "params": SRParamsOptimized[timecostParam][dataset], "score": predictions[index]
#                 }, inp)

#         print("Done with time cost = ", timecostParam)


# with open('experiments/experiment_SR_approach/SROptimalParams.pkl', 'wb') as inp:
#     pickle.dump(SRParamsOptimized, inp)




list_experiments = []

for param_time in timeParams:
    for dataset in datasets:
        
        with open("experiments/experiment_SR_approach/" + str(param_time) +"," + dataset + ".pkl", "rb") as f:
            params = pickle.load(f)["params"]
            # print(params)
        list_experiments.append({
            "dataset": dataset,
            "param_time": param_time,
            "SRparams": params,
        })

#  args = SRParams, sampling_ratio, folderRealData, dataset, timecostParam
args = [(d["SRparams"], sampling_ratio, folderRealData, d["dataset"], d["param_time"]) for d in list_experiments ]
print("Ready to parallelize")
test_scores_results  = Parallel(n_jobs=nb_core)(delayed(predictSR)(func_arg) for func_arg in args)

output_dict = {
    str(d["param_time"]) + ',' + d["dataset"] : d["test_score"] for d in test_scores_results
}


print(output_dict)

with open("test_scores_SR.json", "w") as g:
    json.dump(output_dict, g)
