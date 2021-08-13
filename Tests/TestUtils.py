"""
MULTICLASS
"""

import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import pandas as pd
import numpy as np
import os.path as op
from Economy_K import Economy_K
# from Economy_K_REV import Economy_K_REV
from Economy_Gamma import Economy_Gamma
from Economy_Gamma_Lite import Economy_Gamma_Lite
# from Economy_Gamma_MC_REV_3 import Economy_Gamma_MC_REV_3
# from Economy_K_multiClustering import Economy_K_multiClustering
from Economy_Gamma_MC import Economy_Gamma_MC
from Economy_K_MC import Economy_K_MC
from Economy_Gamma_MC_C1 import Economy_Gamma_MC_C1
from Economy_Gamma_MC_C1_Norm import Economy_Gamma_MC_C1_Norm
try:
    import cPickle as pickle
except ImportError:
    import pickle
import json
from sklearn.metrics import cohen_kappa_score
from sklearn.base import clone
from sklearn.model_selection import StratifiedShuffleSplit
import time
from datetime import datetime


def buildModelName(vp):
    name =  vp["method"] + ',' + vp["dataset"]+ ',' + str(vp["n_groups"])
    if vp["method"] == "Gamma_MC":
        name += ',' + vp["aggregate"]
    return name

def buildExperimentName(vp):
    name =   vp["method"] + ',' + vp["dataset"]+ ',' + str(vp["n_groups"]) + ',' + str(vp["param_time"]) 
    if vp["method"] == "Gamma_MC":
        name += ',' + vp["aggregate"]
    return name

def buildBestOfGroupsName(vp):
    name =  vp["method"] + ',' + vp["dataset"] + ',' + str(vp["param_time"])
    if vp["method"] == "Gamma_MC":
        name += ',' + vp["aggregate"]
    return name

def buildMethodName(vp):
    name = vp["method"]
    if vp["method"] == "Gamma_MC":
        name += ',' + vp["aggregate"]
    return name

def saveRealClassifiers(arguments):
    pathToClassifiers, folderRealData, dataset, classifier = arguments

    # path to data
    filepathTrain = folderRealData + '/' + dataset + '/' + dataset + '_TRAIN.tsv'

    # read data
    train = pd.DataFrame.from_csv(filepathTrain, sep='\t', header=None, index_col=None)
    mn = np.min(np.unique(train.iloc[:,0].values))

    train.iloc[:,0] = train.iloc[:,0].apply(lambda e: 0 if e==mn else 1)

    # get X and y
    Y_train = train.iloc[:, 0]
    X_train = train.loc[:, train.columns != train.columns[0]]


    max_t = X_train.shape[1]
    classifiers = {}

    ## Train classifiers for each time step
    for t in range(1, max_t+1):

        # use the same type classifier for each time step
        classifier_t = clone(classifier)
        # fit the classifier
        classifier_t.fit(X_train.iloc[:, :t], Y_train)
        # save it in memory
        classifiers[t] = classifier_t

    # save the model
    with open(pathToClassifiers + '/classifier'+dataset+'.pkl', 'wb') as output:
        pickle.dump(classifiers, output)

def saveRealModelsECONOMY(cp, vp):
    start_time = time.time()

    use_complete_ECO_model, pathECOmodel, sampling_ratio, orderGamma, ratioVal, pathToRealModelsECONOMY, pathToClassifiers, folderRealData, Cm, min_t, classifier, fears, feat = cp
    method = vp["method"]
    dataset = vp["dataset"]
    group = vp["n_groups"]

    # model name
    modelName = buildModelName(vp)
    pathECOmodel = pathECOmodel + modelName+'.pkl'

    print("Beginning:",modelName)

    if not (os.path.exists(pathToRealModelsECONOMY + '/' + modelName + '.pkl')):
        # read data
        train_classifs = pd.read_csv(op.join(folderRealData, dataset, dataset + '_TRAIN_CLASSIFIERS.tsv'), sep='\t', header=None, index_col=None, engine='python')
        estimate_probas = pd.read_csv(op.join(folderRealData, dataset, dataset + '_ESTIMATE_PROBAS.tsv'), sep='\t', header=None, index_col=None, engine='python')

        mn = np.min(np.unique(train_classifs.iloc[:,0].values))
        mx_t = train_classifs.shape[1] - 1
        nbCLasses = len(train_classifs.iloc[:, 0].unique())
        misClassificationCost = (np.ones((nbCLasses,nbCLasses)) - np.eye(nbCLasses) )*Cm

        # time cost
        timeCost = 0.01 * np.arange(mx_t+1) # arbitrary value

        # choose the method
        if (method == 'Gamma'):
            model = Economy_Gamma(misClassificationCost, timeCost, min_t, classifier, group, orderGamma, sampling_ratio, use_complete_ECO_model, pathECOmodel, fears, dataset, feat, folderRealData)
        elif (method == 'K') :
            model = Economy_K(misClassificationCost, timeCost, min_t, classifier, group, sampling_ratio, use_complete_ECO_model, pathECOmodel, fears, dataset, feat, folderRealData)
        elif (method == 'Gamma_lite'):
            model = Economy_Gamma_Lite(misClassificationCost, timeCost, min_t, classifier, group, orderGamma, sampling_ratio, use_complete_ECO_model, pathECOmodel, fears, dataset, feat)
        elif (method == 'Gamma_MC'):
            aggregate = vp["aggregate"]
            model = Economy_Gamma_MC(misClassificationCost, timeCost, min_t, classifier, group, orderGamma, sampling_ratio, use_complete_ECO_model, pathECOmodel, fears, dataset, feat, folderRealData, aggregate)
        elif (method == 'Gamma_MC_C1'):
            model = Economy_Gamma_MC_C1(misClassificationCost, timeCost, min_t, classifier, group, orderGamma, sampling_ratio, use_complete_ECO_model, pathECOmodel, fears, dataset, feat, folderRealData)
        elif (method == 'Gamma_MC_C1_Norm'):
            model = Economy_Gamma_MC_C1_Norm(misClassificationCost, timeCost, min_t, classifier, group, orderGamma, sampling_ratio, use_complete_ECO_model, pathECOmodel, fears, dataset, feat, folderRealData)
        elif (method == 'K_MC'):
            model = Economy_K_MC(misClassificationCost, timeCost, min_t, classifier, group, sampling_ratio, use_complete_ECO_model, pathECOmodel, fears, dataset, feat, folderRealData)
        else:
            model = Economy_K_multiClustering(misClassificationCost, timeCost, min_t, classifier, group, sampling_ratio, use_complete_ECO_model, pathECOmodel, fears, dataset, feat)
        
        print("CREATED")
        
        # fit the model

        pathToClassifiers = pathToClassifiers + 'classifier' + dataset
        try:
            model.fit(train_classifs, estimate_probas, ratioVal, pathToClassifiers)
        except Exception as e:
            print("EXCEPTION", e)
            return
        print("FITTED")
        # save the model
        with open(pathToRealModelsECONOMY + '/' + modelName + '.pkl', 'wb') as output:
            pickle.dump(model, output)
    print("Done:", modelName,  '-'*(80-len(modelName)), datetime.now(), "in %s s" % (time.time() -start_time) )

def transform_to_format_fears(X):
    nbObs, length = X.shape
    for i in range(nbObs):
        ts = X.iloc[i,:]
        data = {'id':[i for _ in range(length)], 'timestamp':[k for k in range(1,length+1)], 'dim_X':list(ts.values)}
        if i==0:
            df = pd.DataFrame(data)
        else:
            df = df.append(pd.DataFrame(data))
    df = df.reset_index(drop=True)
    return df

def score(model, X_test, y_test, uc, C_m, sampling_ratio, val=None, method='autre', min_t=4, max_t=50):


    # print("DEBUG",  "SCORE FUNCTION ***********************************************************************", model.timeCost)

    nb_observations, _ = X_test.shape
    score_computed = 0
    step = int(max_t*sampling_ratio) if int(max_t*sampling_ratio)>0 else 1
    start = step
    # We predict for every time series [label, tau*]
    if not method=='K':
        if val:
            with open(op.join(model.folderRealData, model.dataset, 'val_preds.pkl') ,'rb') as inp:
                donnes_pred = pickle.load(inp)
            with open(op.join(model.folderRealData, model.dataset, 'val_probas.pkl') ,'rb') as inp:
                donnes_proba = pickle.load(inp)
            # if uc:
            #     # MOD
            #     # with open(op.join(model.folderRealData, 'uc', model.dataset+'val_probas.pkl') ,'rb') as inp:
            #     with open(op.join(model.folderRealData,  model.dataset,'val_probas.pkl') ,'rb') as inp:
            #         donnes_uc = pickle.load(inp)
        if not val:
            with open(op.join(model.folderRealData, model.dataset, 'test_preds.pkl') ,'rb') as inp:
                donnes_pred = pickle.load(inp)
            with open(op.join(model.folderRealData, model.dataset, 'test_probas.pkl') ,'rb') as inp:
                donnes_proba = pickle.load(inp)
            # if uc:
            #     # MOD
            #     # with open(op.join(model.folderRealData, 'uc', model.dataset+'test_uc.pkl') ,'rb') as inp:
            #     with open(op.join(model.folderRealData, model.dataset ,'test_probas.pkl') ,'rb') as inp:
            #         donnes_uc = pickle.load(inp)
    else:
        if val:
            with open(op.join(model.folderRealData, model.dataset, 'val_preds.pkl') ,'rb') as inp:
                donnes_pred = pickle.load(inp)
        else:
            with open(op.join(model.folderRealData, model.dataset, 'test_preds.pkl') ,'rb') as inp:
                donnes_pred = pickle.load(inp)
     
    
    # # TODO: normalize probas
    # if method == "Gamma_MC_C1" and False:
    #     for time in donnes_proba.keys():
    #         donnes_proba[time] = model.proba_scaler.transform(donnes_proba[time])


    

    for i in range(nb_observations):
        # print("DEBUG **********************************************************", "OBSERVATION number", i)
        # The approach is non-myopic, for each time step we predict the optimal
        # time to make the prediction in the future.

        for t in model.timestamps:
            # first t values of x

            x = np.array(list(X_test.iloc[i, :t]))
            if not method=='K':
                # if uc:
                #     pb = donnes_uc[t][i]
                # else:
                #     pb = donnes_proba[t][i]
                pb = donnes_proba[t][i]
            # compute cost of future timesteps (max_t - t)
            if method=='K':
                send_alert, cost = model.forecastExpectedCost(x)
            else:
                send_alert, cost, cst = model.forecastExpectedCost(x,pb)
            #compute tau*
            # predict the label of our time series when tau* = 0 or when we
            # reach max_t
            if (send_alert):
                if model.fears:
                    prediction = model.classifiers[t].predict(transform_to_format_fears(x.reshape(1, -1)))[0]
                elif model.feat:
                    prediction = donnes_pred[t][i]
                else:
                    prediction = model.classifiers[t].predict(x.reshape(1, -1))[0]


                if (prediction != y_test.iloc[i]):
                    score_computed += model.timeCost[t] + C_m
                else:
                    score_computed += model.timeCost[t]
                break
    return (score_computed/nb_observations)

def score_post_optimal(model, X_test, y_test, C_m, sampling_ratio, val=None, min_t=4, max_t=50):

    nb_observations, _ = X_test.shape
    score_computed = 0
    step = int(max_t*sampling_ratio) if int(max_t*sampling_ratio)>0 else 1
    start = step

    if val:
        with open('RealData/'+model.dataset+'/val_preds'+'.pkl' ,'rb') as inp:
            donnes_pred = pickle.load(inp)
        with open('RealData/'+model.dataset+'/val_probas'+'.pkl' ,'rb') as inp:
            donnes_proba = pickle.load(inp)
    if not val:
        with open('RealData/'+model.dataset+'/test_preds'+'.pkl' ,'rb') as inp:
            donnes_pred = pickle.load(inp)
        with open('RealData/'+model.dataset+'/test_probas'+'.pkl' ,'rb') as inp:
            donnes_proba = pickle.load(inp)

    # We predict for every time series [label, tau*]
    for i in range(nb_observations):
        post_costs = []
        timestamps_pred = []
        
        for t in range(start, max_t+1, step):


            x = np.array(list(X_test.iloc[i, :t]))

            pb = donnes_proba[t][i]
            # compute cost of future timesteps (max_t - t)
            _, cost = model.forecastExpectedCost(x,pb)
            post_costs.append(cost)
            timestamps_pred.append(t)
            #compute tau*
            # predict the label of our time series when tau* = 0 or when we
            # reach max_t
        tau_post_star = timestamps_pred[np.argmin(post_costs)]

        if model.fears:
            prediction = model.classifiers[t].predict(transform_to_format_fears(x.reshape(1, -1)))[0]
        elif model.feat:
            prediction = donnes_pred[t][i]
        else:
            prediction = model.classifiers[t].predict(x.reshape(1, -1))[0]

        if (prediction != y_test.iloc[i]):
            score_computed += model.timeCost[tau_post_star] + C_m
        else:
            score_computed += model.timeCost[tau_post_star]
 

    return (score_computed/nb_observations)

def computeScores(arguments, cp, vp, validation):
    start_time = time.time()
    # score_chosen, normalizeTime, C_m, orderGamma, pathToSaveScores, pathToRealModelsECONOMY, folderRealData, method, dataset, group, misClassificationCost, min_t, paramTime = arguments
    score_chosen, normalizeTime, C_m, orderGamma, pathToSaveScores, pathToRealModelsECONOMY, folderRealData, misClassificationCost, min_t = arguments
    
    method = vp["method"]
    param_time = vp["param_time"]
    dataset = vp["dataset"]
    
    # model name
    modelName = buildModelName(vp)
    fullModelName = buildExperimentName(vp)

    # Load model
    try:
        # print("Ready to load model ", op.join(pathToRealModelsECONOMY, modelName + '.pkl'))
        with open(op.join(pathToRealModelsECONOMY, modelName + '.pkl'), 'rb') as input:
            model = pickle.load(input)
    except:
        print("Error loading the model", modelName, " in ", pathToRealModelsECONOMY)
        return

    # paths
    if validation:
        path_score = op.join(pathToSaveScores, 'score'+ fullModelName + '.json')
        filepathTest = op.join(folderRealData, dataset, dataset + '_VAL_SCORE.tsv')
    else:
        path_score = op.join(pathToSaveScores, 'EVALscore'+fullModelName+'.json')
        path_score_post = op.join(pathToSaveScores,'EVALscorePOST'+fullModelName+'.json')
        # path to data
        filepathTest = op.join(folderRealData, dataset, dataset+'_TEST_SCORE.tsv')
        

    if not (os.path.exists(path_score)):
        print("NOT FOUND: ", path_score)
        # read data
        val = pd.read_csv(filepathTest, sep='\t', header=None, index_col=None, engine='python')
        mn = np.min(np.unique(val.iloc[:,0].values))
 
        # get X and y
        y_val = val.iloc[:, 0]
        X_val = val.loc[:, val.columns != val.columns[0]]
        mx_t = X_val.shape[1]

        # choose the method
        try:
            print("Computing", fullModelName)

            timeCostt = float(vp["param_time"]) * np.arange(model.timestamps[-1]+1)            
            if normalizeTime:
                timeCostt *= (1/mx_t)     
            setattr(model, 'timeCost', timeCostt)
        
        except pickle.UnpicklingError:
            print('PROOOOBLEM', modelName)

        if score_chosen == 'star':
            if method =='Gamma_MC':
                aggregate = vp["aggregate"]
                setattr(model,"aggregate", aggregate)
                score_model = score(model, X_val, y_val, True, C_m, model.sampling_ratio, val=validation, method='autre', min_t=1, max_t=50)
            else:
                score_model = score(model, X_val, y_val, False, C_m, model.sampling_ratio, val=validation, method=method, min_t=1, max_t=50)
        if score_chosen == 'post':
            score_model = score_post_optimal(model, X_val, y_val, C_m, model.sampling_ratio, max_t=mx_t)

        with open(path_score, 'w') as outfile:
            json.dump({fullModelName:score_model}, outfile)
    else:
        if score_chosen == 'star':
            with open(path_score) as f:
                try:
                    loadedjson = json.loads(f.read())
                except:
                    print('BUUUUUUUUUUUUUUUUUUUUUG',fullModelName)
        else:
            with open(path_score_post) as f:
                try:
                    loadedjson = json.loads(f.read())
                except:
                    print('BUUUUUUUUUUUUUUUUUUUUUG',fullModelName)
        score_model = list(loadedjson.values())[0]    


    # return (fullModelName,score_model)
    # We return directly the dictionary of variable parameters to characterize the model²

    print("Score for", fullModelName,  '-'*(80-len(fullModelName)), datetime.now(), "in %s s" % (time.time() -start_time), "validation:", validation )
    return (vp,score_model)

def computeBestGroup(vps, modelName_score, INF, methods):
    
    bestGroup = {buildBestOfGroupsName(variant):[1,INF] for variant in vps}
    
    print(modelName_score)
    print(bestGroup)

    for (params, score_model) in modelName_score:
        # method, dataset, group, paramTime, _aggregate = modelName.split(',')
        # method, dataset, group, paramTime, _aggregate = params["method"], params["dataset"], params["n_groups"], params["param_time"]
        
        
        # group = int(group)
        # paramTime = float(paramTime)
        # if paramTime == 1.0:
        #     paramTime = int(paramTime)

        group_name = buildBestOfGroupsName(params)

        if (bestGroup[group_name][1] > score_model):
            bestGroup[group_name][0] = int(params["n_groups"])
            bestGroup[group_name][1] = score_model
    print(bestGroup)
    return bestGroup


def scoreMori(arguments):

    moriParams, timecostParam, ep_probas, ep_preds, val_probas, val_preds, y_test_ep, y_test_val, nb_observations_val, nb_observations_ep, timestamps, timecost, max_t = arguments

    score = 0
    for i in range(nb_observations_ep):

        for t in timestamps:

            proba1 = ep_probas[t][i]
            proba2 = 1.0 - proba1
            if proba1 > proba2:
                maxiProba = proba1
                scndProba = proba2
            else:
                maxiProba = proba2
                scndProba = proba1

            # Stopping rule
            sr = moriParams[0] * maxiProba + moriParams[1] * (maxiProba-scndProba) + moriParams[2] * (t / max_t)

            if sr > 0 or t==timestamps[-1]:
                
                if y_test_ep.iloc[i] == ep_preds[t][i]:
                    score += timecost[t]
                else:
                    score += timecost[t] + 1 #C_m = 1
                break

    for i in range(nb_observations_val):

        for t in timestamps:

            proba1 = val_probas[t][i]
            proba2 = 1.0 - proba1
            if proba1 > proba2:
                maxiProba = proba1
                scndProba = proba2
            else:
                maxiProba = proba2
                scndProba = proba1

            # Stopping rule
            sr = moriParams[0] * maxiProba + moriParams[1] * (maxiProba-scndProba) + moriParams[2] * (t / max_t)

            if sr > 0 or t==timestamps[-1]:
                
                if y_test_val.iloc[i] == val_preds[t][i]:
                    score += timecost[t]
                else:
                    score += timecost[t] + 1 #C_m = 1
                break
    print('------------FINISH---------- : ', timecostParam)
    return score / (nb_observations_ep + nb_observations_val)

def predictMori(arguments):
    print('DEBUT !! ')

    moriParams, sampling_ratio, folderRealData, dataset, timecostParam, pathSortie = arguments

    if not (os.path.exists(pathSortie+'/'+dataset+str(timecostParam)+'.pkl')):

        with open(folderRealData+'/'+dataset+'/test_probas.pkl' ,'rb') as inp:
            test_probas = pickle.load(inp)
        with open(folderRealData+'/'+dataset+'/test_preds.pkl' ,'rb') as inp:
            test_preds = pickle.load(inp)

        # compute AvgCost
        filepathTest = folderRealData+'/'+dataset+'/'+dataset+'_TEST_SCORE.tsv'
        test = pd.read_csv(filepathTest, sep='\t', header=None, index_col=None, engine='python')
        li, col = test.shape
        max_t = col-1
        step = int(max_t*sampling_ratio) if int(max_t*sampling_ratio)>0 else 1
        timestamps = [t for t in range(step, max_t + 1, step)]
        mn = np.min(np.unique(test.iloc[:,0].values))
        timecost = timecostParam * np.arange(max_t+1)
        test.iloc[:,0] = test.iloc[:,0].apply(lambda e: 0 if e==mn else 1)
        # get X and y
        y_test = test.iloc[:, 0]
        del test
        nb_observations=li
        
        # compute trigger times
        trigger_times = []
        score = 0
        preds = []
        for i in range(nb_observations):

            for t in timestamps:

                proba1 = test_probas[t][i]
                proba2 = 1.0 - proba1
                if proba1 > proba2:
                    maxiProba = proba1
                    scndProba = proba2
                else:
                    maxiProba = proba2
                    scndProba = proba1

                # Stopping rule
                sr = moriParams[0] * maxiProba + moriParams[1] * (maxiProba-scndProba) + moriParams[2] * (t / max_t)

                if sr > 0 or t==timestamps[-1]:
                    preds.append(test_preds[t][i])
                    trigger_times.append(t)
                    if y_test.iloc[i] == test_preds[t][i]:
                        score += timecost[t]
                    else:
                        score += timecost[t] + 1 #C_m = 1
                    break
        score = score/nb_observations
        print('score : ', score)
        
        kapa = round(cohen_kappa_score(y_test.values.tolist(), preds),2) 
        print('kapa : ', kapa)
        with open(pathSortie+'/'+dataset+str(timecostParam)+'.pkl', 'wb') as outfile:
            pickle.dump([trigger_times, score, preds, kapa, timecostParam, dataset], outfile)
        colms = ['Dataset', 'timeParam', 'Score', 'Med_tau', 'Kappa']
        v = [dataset, timecostParam, score, np.median(np.array(trigger_times)), kapa]
        
        return {key:value for key, value in zip(colms, v)}

def computePredictionsEconomy(cst_params, vp):
    normalizeTime, pathToSavePredictions, pathToIntermediateResults, folderRealData, pathToRealModels = cst_params
    dataset = vp["dataset"]
    method = vp["method"]
    fullModelName = buildExperimentName(vp)
    modelName = buildModelName(vp)
    paramTime = float(vp["param_time"])

    # path to data
    filepathTest = op.join(folderRealData,dataset,dataset+'_TEST_SCORE.tsv')

    # modelName = buildExperimentName(vp)

    if not (os.path.exists(op.join(pathToSavePredictions,'PREDECO'+fullModelName+'.pkl'))):
        print("not found: ", pathToSavePredictions,'PREDECO'+fullModelName+'.pkl') 
        # read data
        test = pd.read_csv(filepathTest, sep='\t', header=None, index_col=None, engine='python')
        mn = np.min(np.unique(test.iloc[:,0].values))

        # get X and y
        y_test = test.iloc[:, 0]
        X_test = test.loc[:, test.columns != test.columns[0]]
        mx_t = X_test.shape[1]

        with open(op.join(pathToRealModels, modelName + '.pkl'), 'rb') as input:
            model = pickle.load(input)
            timeCostt = paramTime * np.arange(model.timestamps[-1]+1)
            if normalizeTime:
                timeCostt *= (1/mx_t)
            
            setattr(model, 'timeCost', timeCostt)

        with open(op.join(folderRealData,dataset,'test_probas.pkl') ,'rb') as inp:
            test_probas = pickle.load(inp)
        with open(op.join(folderRealData,dataset,'test_preds.pkl'),'rb') as inp:
            test_preds = pickle.load(inp)


        # if method=='Gamma_MC':
        #     with open(op.join(folderRealData,dataset,'test_probas.pkl'),'rb') as inp:
        #         test_uc = pickle.load(inp)
        #     preds_tau = model.predict(X_test, oneIDV=False, donnes=[test_uc, test_preds])
        # else:
        preds_tau = model.predict(X_test, oneIDV=False, donnes=[test_probas, test_preds])

        #preds_post = model.predict_post_tau_stars(X_test, [test_probas, test_preds])
        #preds_optimal = model.predict_optimal_algo(X_test, y_test, test_preds)
        #metric = preds_tau, preds_post, preds_optimal

        with open(op.join(pathToSavePredictions, 'PREDECO'+fullModelName+'.pkl'), 'wb') as outfile:
            pickle.dump({modelName: preds_tau}, outfile)

        return preds_tau, modelName
    else:
        # print('hello')
        with open(op.join(pathToSavePredictions, 'PREDECO'+fullModelName+'.pkl'), 'rb') as outfile:
            loadedjson = pickle.load(outfile)
            
        return list(loadedjson.values())[0], list(loadedjson.keys())[0]

def computeMetricsNeeded(arguments):

    sampling_ratio = arguments[0]
    preds_tau, preds_post, preds_optimal, modelName = arguments[1]
    pathToIntermediateResults = arguments[2]
    folderRealData = arguments[3]

    method, dataset, group, timeparam = modelName.split(',')

    organized_metrics = {dataset+','+method:[]}
    tau_et = np.array(preds_tau)[:,0]
    tau_post = np.array(preds_post)[:,0]

    #print('taille tau_et : ', tau_et.shape)

    tau_opt = np.array(preds_optimal)[:,0]
    #print(modelName)
    #print('taille tau_post : ', tau_post.shape)

    f_et = np.array(preds_tau)[:,1]

    f_post = np.array(preds_post)[:,1]

    f_opt = np.array(preds_optimal)[:,1]

    objectives = [tau_et, tau_post, tau_opt, f_et, f_post, f_opt]
    organized_metrics[dataset+','+method].append(timeparam)
    for e in objectives:
        organized_metrics[dataset+','+method].append(np.mean(e))
        organized_metrics[dataset+','+method].append(np.std(e))

    for e1, e2 in zip(objectives[1:3], objectives[4:]):
        organized_metrics[dataset+','+method].append(np.mean(abs(tau_et-e1)))
        organized_metrics[dataset+','+method].append(np.std(abs(tau_et-e1)))
        organized_metrics[dataset+','+method].append(np.mean(abs(f_et-e2)))
        organized_metrics[dataset+','+method].append(np.std(abs(f_et-e2)))

    organized_metrics[dataset+','+method].append(group) #group



    # AUC import packages ???
    filepathTest = folderRealData+'/'+dataset+'/'+dataset+'_TEST_SCORE.tsv'
    test = pd.read_csv(filepathTest, sep='\t', header=None, index_col=None, engine='python')
    mn = np.min(np.unique(test.iloc[:,0].values))
    test.iloc[:,0] = test.iloc[:,0].apply(lambda e: 0 if e==mn else 1)
    y_test = test.iloc[:, 0]
    X_test = test.loc[:, test.columns != test.columns[0]]
    max_t = X_test.shape[1]
    # pourcentage des prediction à min_t
    min_t = int(max_t*sampling_ratio) if int(max_t*sampling_ratio)>0 else 1

    for i in range(min_t, max_t+1, min_t):
        mx_t = i
    count_min_t = 0
    count_max_t = 0
    for elem in tau_et:
        if (elem == min_t):
            count_min_t += 1
        if (elem == mx_t):
            count_max_t += 1

    count_min_t /= len(tau_et)
    count_max_t /= len(tau_et)
    organized_metrics[dataset+','+method].append(count_min_t)
    organized_metrics[dataset+','+method].append(count_max_t)

    print(np.array(preds_tau)[:,2])
    kappa_tau = round(cohen_kappa_score(y_test.values.tolist(), list(map(int, np.array(preds_tau)[:,2]))),2)
    kappa_opt = round(cohen_kappa_score(y_test.values.tolist(), list(map(int, np.array(preds_post)[:,2]))),2)
    kappa_post = round(cohen_kappa_score(y_test.values.tolist(), list(map(int, np.array(preds_optimal)[:,2]))),2)

    organized_metrics[dataset+','+method].append(kappa_tau)
    organized_metrics[dataset+','+method].append(kappa_post)
    organized_metrics[dataset+','+method].append(kappa_opt)


    # pourcentage des cas ou on est précoce au post optimal
    counters = [0,0,0,0]
    for tauu, tau_ppost, tau_opti in zip(tau_et, tau_post, tau_opt):
        if tauu < tau_ppost:
            counters[0] = counters[0] + 1
        if tauu < tau_opti:
            counters[1] = counters[1] + 1
        if tauu >= tau_ppost:
            counters[2] = counters[2] + 1
        if tauu >= tau_opti:
            counters[3] = counters[3] + 1
    for i in range(4):
        counters[i] = round(counters[i] / len(tau_et),2)

    organized_metrics[dataset+','+method].append(counters[0])
    organized_metrics[dataset+','+method].append(counters[1])
    organized_metrics[dataset+','+method].append(counters[2])
    organized_metrics[dataset+','+method].append(counters[3])

    # compute medians tau_et, tau_post, tau_opt, f_et, f_post, f_opt
    organized_metrics[dataset+','+method].append(np.median(tau_et))
    organized_metrics[dataset+','+method].append(np.median(tau_post))
    organized_metrics[dataset+','+method].append(np.median(tau_opt))
    organized_metrics[dataset+','+method].append(np.median(f_et))
    organized_metrics[dataset+','+method].append(np.median(f_post))
    organized_metrics[dataset+','+method].append(np.median(f_opt))

    # compute medians differences
    organized_metrics[dataset+','+method].append(np.median(abs(tau_et-tau_post)))
    organized_metrics[dataset+','+method].append(np.median(abs(f_et-f_post)))




    with open(pathToIntermediateResults+'/MetricsNeeded'+modelName+'.pkl', 'wb') as outfile:
        pickle.dump(organized_metrics, outfile)

    return organized_metrics
