
#########################################################################################
# print('###################### yachench, gsdgrdg @Orange labs ######################')
#########################################################################################

"""
EXPERIMENT FOR MULTICLASS VERSIONS OF ECONOMY

In every part, an experiment is represented by a dictionary with the experiemnt specific paramters (dataset, nb_groups, ...)

"""

import multiprocessing
from TestUtils import *
import os.path as op
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed
import json
import argparse
import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.path.pardir)))
# from TestUtils import saveRealClassifiers, saveRealModelsECONOMY, score, computeScores, computeBestGroup, evaluate, computePredictionsEconomy, computeMetricsNeeded
try:
    import cPickle as pickle
except ImportError:
    import pickle
# "Gamma_lite", "Gamma", "K",
if __name__ == '__main__':

    #########################################################################################
    print('############################### TODOS ##################################')
    #########################################################################################
    # todo :
    #        - meme clustering pour les deux méthodes
    #        -  pourcentage max_t
    #        - AUC n'a pas de sens (nous n'avons pas le même classifieur pour tous les individus calibration modèle) remplacer l'auc par kappa
    #        - evaluer les classifieurs à chaque pas de temps, (choix de min_t selon la perf des classifieurs) evolution de l'auc moyenne par rapport au temps
    #        - score à rajouter sur les métriques
    #        - il faut apprendre economy models pour tous les timestamps et après pour chaque sampling pouvoir les utiliser

    #########################################################################################
    print('############################### PARAMS CONFIG ##################################')
    #########################################################################################
    # command line
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--pathToInputParams',
                        help='path to json file with input params', required=True)
    parser.add_argument('--pathToOutputs', help='path outputs', required=True)

    args = parser.parse_args()
    print(args)
    pathToParams = args.pathToInputParams

    # load input params
    with open(pathToParams) as configfile:
        configParams = json.load(configfile)
        print(configParams)
    
    # set variables
    nbDatasets = configParams['nbDatasets']
    folderRealData = configParams['folderRealData']
    sampling_ratio = configParams['sampling_ratio']
    # et summary and sort by train set size
    classifier = LogisticRegression(
        random_state=0, solver='lbfgs', multi_class='multinomial')
    timeParams = configParams['timeParams']
    
    # Every group
    # nbGroups = np.arange(1, configParams['nbGroups']) 
    nbGroups = [n for n in range(1, configParams['nbGroups'] + 1)]

    # We add the number of groups equal to the number of classes and the double of the number of classes
    # for nb_classes in np.sort(np.array(list(configParams["datasetsNbClasses"].values()))):
    #     if nb_classes >= configParams["nbGroups"]:
    #         nbGroups.append(nb_classes)
    #     if nb_classes*2 >= configParams["nbGroups"]:
    #         nbGroups.append(2*nb_classes)
    # nbGroups = np.unique(nbGroups)

    print(nbGroups)

    # nbGroups = range(14, configParams['nbGroups'])

    methods = configParams['methods']
    C_m = configParams['misClassificationCost']
    misClassificationCost = C_m
    min_t = configParams['min_t']
    pathToClassifiers = configParams['pathToClassifiers']
    allECOmodelsAvailable = configParams['allECOmodelsAvailable']
    # nb_core = 30  # multiprocessing.cpu_count()
    nb_core = configParams["nbCore"]
    orderGamma = configParams['orderGamma']
    ratioVal = configParams['ratioVal']
    pathToIntermediateResults = configParams['pathToIntermediateResults']
    pathToResults = configParams['pathToResults']
    saveClassifiers = configParams['saveClassifiers']
    pathToRealModelsECONOMY = configParams['pathToRealModelsECONOMY']
    pathToSaveScores = configParams['pathToSaveScores']
    pathToSavePredictions = configParams['pathToSavePredictions']
    normalizeTime = configParams['normalizeTime']
    use_complete_ECO_model = configParams['use_complete_ECO_model']
    pathECOmodel = configParams['pathECOmodel']
    fears = configParams['fears']
    score_chosen = configParams['score_chosen']
    feat = configParams['feat']
    datasets = configParams['Datasets']
    INF = 10000000
    aggregates = configParams['aggregates']



    ########################################################################################
    print('################################ SAVE ECONOMY ##################################')
    ########################################################################################

    variable_parameters_list_save_models = []
    for dataset in datasets:
        additional_groups_for_dataset = []
        nb_classes = configParams["datasetsNbClasses"][dataset]
        if nb_classes >= configParams["nbGroups"]:
            additional_groups_for_dataset.append(nb_classes)
        if nb_classes*2 >= configParams["nbGroups"]:
            additional_groups_for_dataset.append(2*nb_classes)

        for n_groups in nbGroups + additional_groups_for_dataset :
            for method in methods:
                base_parameters = {
                        "method": method,
                        "dataset": dataset,
                        "n_groups": n_groups,
                    }
                if (method == "Gamma_MC"):
                    for aggregate in aggregates:
                        variable_parameters_list_save_models.append(
                            {**base_parameters, "aggregate": aggregate})
                else:
                    variable_parameters_list_save_models.append(base_parameters)

    print(variable_parameters_list_save_models)


    if not allECOmodelsAvailable:
        cst_args = (use_complete_ECO_model, pathECOmodel, sampling_ratio, orderGamma, ratioVal, pathToRealModelsECONOMY,
                    pathToClassifiers, folderRealData, misClassificationCost, min_t, classifier, fears, feat)
        Parallel(n_jobs=nb_core)(delayed(saveRealModelsECONOMY)(cst_args, vp)
                                 for vp in variable_parameters_list_save_models)


    #########################################################################################
    print('############################### Compute scores #################################')
    #########################################################################################

    # We add aggregate parameters if method if Gamma_MC only
    variable_parameters_list = []
    for dataset in datasets:

        additional_groups_for_dataset = []
        nb_classes = configParams["datasetsNbClasses"][dataset]
        if nb_classes >= configParams["nbGroups"]:
            additional_groups_for_dataset.append(nb_classes)
        if nb_classes*2 >= configParams["nbGroups"]:
            additional_groups_for_dataset.append(2*nb_classes)

        for n_groups in nbGroups + additional_groups_for_dataset :
            for param_time in timeParams:
                for method in methods:
                    base_parameters = {
                            "method": method,
                            "dataset": dataset,
                            "n_groups": n_groups,
                            "param_time": param_time,
                        }
                    if (method == "Gamma_MC"):
                        for aggregate in aggregates:
                            variable_parameters_list.append(
                                {**base_parameters, "aggregate": aggregate})
                    else:
                        variable_parameters_list.append(base_parameters)




    print(op.join(pathToIntermediateResults, 'modelName_score.pkl'))
    cst_args = (score_chosen, normalizeTime, C_m, orderGamma, pathToSaveScores,
                pathToRealModelsECONOMY, folderRealData, misClassificationCost, min_t)
    modelName_score = Parallel(n_jobs=nb_core)(delayed(computeScores)(
        cst_args, configParams, variable_parameters, validation = True) for variable_parameters in variable_parameters_list)

    print(modelName_score)

    with open(op.join(pathToIntermediateResults, 'modelName_score.pkl'), 'wb') as outfile:
        pickle.dump(modelName_score, outfile)

    modelName_score = [x for x in modelName_score if x is not None]
    print("Model name score: ", modelName_score)

    #########################################################################################
    print('##################### Compute best hyperparam : nbGroups #######################')
    #########################################################################################
    bestGroup = computeBestGroup(variable_parameters_list, modelName_score, INF, methods)
    with open(op.join(pathToIntermediateResults, 'bestGroup.pkl'), 'wb') as outfile:
        pickle.dump(bestGroup, outfile)
    """with open(op.join(pathToIntermediateResults, 'bestGroup.pkl'), 'rb') as outfile:
        bestGroup = pickle.load(outfile)"""

    #########################################################################################
    print('############################ Compute best scores ###############################')
    #########################################################################################
    # use the best group for evaluating on data test
    # compute scores
    cst_param = (score_chosen, normalizeTime, C_m, orderGamma, pathToSaveScores,
                pathToRealModelsECONOMY, folderRealData, misClassificationCost, min_t)

    variable_parameters_list_best_groups = []
    for dataset in datasets:
        for param_time in timeParams:
            for method in methods:
                base_parameters = {
                    "method": method,
                    "dataset": dataset,
                    "param_time": param_time,
                }
                if (method == "Gamma_MC"):
                    for aggregate in aggregates:
                        variable_parameters_list_best_groups.append({**base_parameters, "aggregate": aggregate})
                else:
                    variable_parameters_list_best_groups.append(base_parameters)
   
   
    # Add the best group depending on the experiment
    for vp in variable_parameters_list_best_groups:
        vp["n_groups"] = bestGroup[buildBestOfGroupsName(vp)][0]

    best_score = Parallel(n_jobs=nb_core)(delayed(computeScores)(
        cst_param, configParams, variant, validation = False) for variant in variable_parameters_list_best_groups)

    with open(op.join(pathToIntermediateResults, 'best_score.pkl'), 'wb') as outfile:
        pickle.dump(best_score, outfile)
    # with open(op.join(pathToIntermediateResults, 'best_score.pkl'), 'rb') as outfile:
    #     best_score = pickle.load(outfile

    #########################################################################################
    print('############################ Compute best scores post opt ###############################')
    #########################################################################################
    # use the best group for evaluating on data test
    # compute scores
    """func_args_best_score_post = []
    for dataset in datasets:
        for paramTime in timeParams:
            for method in methods:
                func_args_best_score_post.append(('post', normalizeTime, C_m, pathToRealModelsECONOMY, folderRealData, dataset, method, bestGroup[method + ',' + dataset + ',' + str(paramTime)][0], paramTime, pathToSaveScores))
    best_score_post = Parallel(n_jobs=nb_core)(delayed(evaluate)(func_arg) for func_arg in func_args_best_score_post)
    with open(pathToIntermediateResults+'/best_score_post.pkl', 'wb') as outfile:
        pickle.dump(best_score_post, outfile)
    with open(pathToIntermediateResults+'/best_score_post.pkl', 'rb') as outfile:
        best_score_post = pickle.load(outfile)"""

    #########################################################################################
    print('############################ Compute best scores ###############################')
    #########################################################################################

    results = {buildExperimentName(vp): score_model for (
        vp, score_model) in best_score}

    with open(op.join(pathToResults, 'results.pkl'), 'wb') as outfile:
        pickle.dump(results, outfile)

    with open(op.join(pathToResults, 'results.json'), 'w') as outfile:
        json.dump(results, outfile)

    """results_post = {}
    for e in best_score_post:
        (modelName, score_model) = e
        results_post[modelName] = score_model
    with open(pathToResults+'/results_post.pkl', 'wb') as outfile:
        pickle.dump(results_post, outfile)
    with open(pathToResults+'/results.pkl', 'rb') as outfile:
        results = pickle.load(outfile)
    with open(pathToResults+'/results_post.pkl', 'rb') as outfile:
        results_post = pickle.load(outfile)"""
    with open(op.join(pathToResults, 'results.pkl'), 'rb') as outfile:
        results = pickle.load(outfile)

    #########################################################################################
    print('############################ Compute predictions ###############################')
    #########################################################################################
    cst_params = (normalizeTime, pathToSavePredictions,
                  pathToIntermediateResults, folderRealData, pathToRealModelsECONOMY)
    predictions = Parallel(n_jobs=nb_core)(delayed(computePredictionsEconomy)(
        cst_params, vp) for vp in variable_parameters_list_best_groups)

    print('FINIIIIIIIIIIIIIISH ############################################################################################"')
    with open(op.join(pathToResults, 'predictions.pkl'), 'wb') as outfile:
        pickle.dump(predictions, outfile)

    # with open(pathToResults+'/predictions.pkl', 'rb') as outfile:
    #    predictions = pickle.load(outfile)

    #########################################################################################
    print('############################## Compute metrics #################################')
    #########################################################################################

    """metrics = Parallel(n_jobs=nb_core)(delayed(computeMetricsNeeded)([ sampling_ratio, func_arg, pathToIntermediateResults,folderRealData]) for func_arg in predictions)
    cols = ['Dataset','Method', 'Score', 'timeParam', 'meanTauStar','stdTauStar','meanTauPost','stdTauPost','meanTauOPT','stdTauOPT', 'mean_f_Star','std_f_Star', 'mean_f_Post','std_f_Post', 'mean_f_Opt','std_f_Opt', 'mean_diff_tauStar_tauPost','std_diff_tauStar_tauPost', 'mean_diff_fStar_fPost','std_diff_fStar_fPost', 'mean_diff_tauStar_tauOpt','std_diff_tauStar_tauOpt', 'mean_diff_fStar_fOpt','std_diff_fStar_fOpt', 'Group', 'pourcentage_min_t', 'pourcentage_max_t', 'kappa_star', 'kappa_post', 'kappa_opt', 'pourcentage_taustar_inf_taupost', 'pourcentage_taustar_inf_tauopt', 'pourcentage_taustar_supstrict_taupost', 'pourcentage_taustar_sustrict_tauopt', 'median_tau_et', 'median_tau_post', 'median_tau_opt', 'median_f_et', 'median_f_post', 'median_f_opt', 'median_diff_tau_et_post', 'median_diff_f_et_post', 'Score_post']
    df_metrics = pd.DataFrame(columns=cols)
    for e in metrics:
        for k,v in e.items():
            dataset, method = k.split(',')
            a = str(v[21])
            b = str(v[0])
            v.insert(0, results[method+','+dataset+','+a+','+b])
            v.insert(0, method)
            v.insert(0, dataset)
            v.append(results_post[method+','+dataset+','+a+','+b])
            df_metrics = df_metrics.append({key:value for key, value in zip(cols, v)}, ignore_index=True)
            print(df_metrics)
    print(df_metrics['Group'])
    with open(pathToResults+'/df_metrics.pkl', 'wb') as outfile:
        pickle.dump(df_metrics, outfile)

    df_metrics.to_html('table.html')"""
