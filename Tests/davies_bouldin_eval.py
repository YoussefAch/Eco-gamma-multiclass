from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import pickle
import os.path
from TestUtils import buildModelName, buildBestOfGroupsName
import matplotlib.pyplot as plt
import numpy as np

"""
    Computes and plots the Davies-Bouldin score for the validation set (TODO: what set ?)
    on the kmeans classifiers in the Gamma _MC_C1 method (Economy gamma with clustering on classifier outputs
    in order to make groups)

"""

# 1) extract kmeans from economy models for each number of groups
# 2) open classifier outputs from files
# 2) apply kmeans and compute score



nbGroups = range(2, 17)
# timeParams = [0.001, 0.01, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5, 1]
timeParams = [0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

datasets_nb_classes = {'Coffee': 2, 'Beef': 5, 'OliveOil': 4, 'Lighting2': 2, 'Lighting7': 7, 'FaceFour': 4, 'ECG200': 2, 'Trace': 4, 'Gun_Point': 2, 'FISH': 7, 'OSULeaf': 6, 'Synthetic_control': 6, 'DiatomSizeReduction': 4, 'Haptics': 5, 'Cricket_X': 12, 'Cricket_Y': 12, 'Cricket_Z': 12, 'Adiac': 37, '50words': 50, 'InlineSkate': 7, 'SonyAIBORobotSurface': 2, 'SwedishLeaf': 15, 'WordsSynonyms': 25, 'MedicalImages': 10, 'ECGFiveDays': 2, 'CBF': 3, 'SonyAIBORobotSurfaceII': 2, 'Symbols': 6, 'ItalyPowerDemand': 2, 'TwoLeadECG': 2, 'MoteStrain': 2, 'CinC_ECG_torso': 4, 'FaceAll': 14, 'NonInvasiveFatalECG_Thorax1': 42, 'NonInvasiveFatalECG_Thorax2': 42, 'FacesUCR': 14, 'MALLAT': 8, 'Yoga': 2, 'StarLightCurves': 3, 'uWaveGestureLibrary_X': 8, 'uWaveGestureLibrary_Y': 8, 'uWaveGestureLibrary_Z': 8, 'ChlorineConcentration': 3, 'Two_Patterns': 4, 'Wafer': 2}
datasets_ep_shapes = {
    "50words" :  (253, 271),
    "Adiac" :  (218, 177),
    "Beef" :  (17, 471),
    "CBF" :  (260, 129),
    "ChlorineConcentration" :  (1206, 167),
    "CinC_ECG_torso" :  (398, 1640),
    "Coffee" :  (16, 287),
    "Cricket_X" :  (218, 301),
    "Cricket_Y" :  (218, 301),
    "Cricket_Z" :  (218, 301),
    "DiatomSizeReduction" :  (90, 346),
    "ECG200" :  (56, 97),
    "ECGFiveDays" :  (247, 137),
    "FaceAll" :  (630, 132),
    "FaceFour" :  (31, 351),
    "FacesUCR" :  (630, 132),
    "FISH" :  (98, 464),
    "Gun_Point" :  (56, 151),
    "Haptics" :  (130, 1093),
    "InlineSkate" :  (182, 1883),
    "ItalyPowerDemand" :  (307, 25),
    "Lighting2" :  (34, 638),
    "Lighting7" :  (40, 320),
    "MALLAT" :  (672, 1025),
    "MedicalImages" :  (319, 100),
    "MoteStrain" :  (356, 85),
    "OliveOil" :  (17, 571),
    "OSULeaf" :  (124, 428),
    "SonyAIBORobotSurface" :  (174, 71),
    "SonyAIBORobotSurfaceII" :  (274, 66),
    "StarLightCurves" :  (1258, 1025),
    "SwedishLeaf" :  (315, 129),
    "Symbols" :  (286, 399),
    "Synthetic_control" :  (168, 61),
    "Trace" :  (56, 276),
    "TwoLeadECG" :  (325, 83),
    "Two_Patterns" :  (1400, 129),
    "uWaveGestureLibrary_X" :  (1254, 316),
    "uWaveGestureLibrary_Y" :  (1254, 316),
    "uWaveGestureLibrary_Z" :  (1254, 316),
    "Wafer" :  (2006, 153),
    "WordsSynonyms" :  (253, 271),
    "Yoga" :  (924, 427),
    "NonInvasiveFatalECG_Thorax1" :  (1054, 751),
    "NonInvasiveFatalECG_Thorax2" :  (1054, 751)
}

# datasets = ["Coffee", "Beef", "OliveOil", "Lighting2", "Lighting7", "FaceFour", "ECG200", "Trace", "Gun_Point", "FISH", "OSULeaf", "Synthetic_control", "DiatomSizeReduction", "Haptics", "Cricket_X", "Cricket_Y", "Cricket_Z", "Adiac", "50words", "InlineSkate", "SonyAIBORobotSurface", "SwedishLeaf", "WordsSynonyms", "MedicalImages", "ECGFiveDays", "CBF",
#             "SonyAIBORobotSurfaceII", "Symbols", "ItalyPowerDemand", "TwoLeadECG", "MoteStrain", "CinC_ECG_torso", "FaceAll", "NonInvasiveFatalECG_Thorax1", "NonInvasiveFatalECG_Thorax2", "FacesUCR", "MALLAT", "Yoga", "StarLightCurves", "uWaveGestureLibrary_X", "uWaveGestureLibrary_Y", "uWaveGestureLibrary_Z", "ChlorineConcentration", "Two_Patterns", "Wafer"]
datasets = ['Beef', 'OliveOil', 'Lighting7', 'FaceFour', 'Trace', 'FISH', 'OSULeaf', 'Synthetic_control', 'DiatomSizeReduction', 'Haptics', 'Cricket_X', 'Cricket_Y', 'Cricket_Z', 'Adiac', '50words', 'InlineSkate', 'SwedishLeaf', 'WordsSynonyms', 'MedicalImages', 'CBF', 'Symbols', 'CinC_ECG_torso', 'FaceAll', 'NonInvasiveFatalECG_Thorax1', 'NonInvasiveFatalECG_Thorax2', 'FacesUCR', 'MALLAT', 'StarLightCurves', 'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y', 'uWaveGestureLibrary_Z', 'ChlorineConcentration', 'Two_Patterns']

# datasets = ["Cricket_X"]
# datasets = ["NonInvasiveFatalECG_Thorax1"]
# datasets = ["OliveOil"]


PATH_FOLDER = "experiments/results tornado/tor_2_exp4"
PATH_EXPERIMENT = PATH_FOLDER +  "/experiment_tor_2"
PATH_DATA = "RealData-MORI"

PATH_FIGURES = os.path.join(PATH_FOLDER, "figures")
if not os.path.exists(PATH_FIGURES):
    os.mkdir(PATH_FIGURES)


with open(os.path.join(PATH_EXPERIMENT, "intermediate_results", "bestGroup.pkl"), "rb") as f:
    best_groups_and_scores = pickle.load(f)

# Get models
# for dataset in datasets:
    # plt.figure()

    # best_groups_per_time_param = {}
    # for param_time in timeParams:

    #     vp_best_group = {"method": "Gamma_MC_C1", "dataset":dataset, "param_time": param_time}
    #     best_groups_per_time_param[param_time] = best_groups_and_scores[buildBestOfGroupsName(vp_best_group)][0]

    # print(best_groups_per_time_param)
    
    # for n_groups in nbGroups:
    #     vp = {"method": "Gamma_MC_C1", "dataset":dataset, "n_groups": n_groups}
    #     with open(os.path.join(PATH_EXPERIMENT, "modelsECONOMY", buildModelName(vp)+ ".pkl" ), 'rb') as f:
    #         model  = pickle.load(f)
    #         clusterings = model.clusterings
            
    #         Y = []
    #         X = list(clusterings.keys())
    #         for t in clusterings.keys():
    #             with open(os.path.join(PATH_DATA, dataset, "test_probas_" + str(t) + ".pkl"), 'rb') as g:
    #                 data = pickle.load(g)
    #                 Y.append(davies_bouldin_score(data, clusterings[t].predict(data)))
    #         # print(X)
    #         # print(Y)
    #         title_appendix = ""
    #         for param_time in timeParams:
    #             if best_groups_per_time_param[param_time] == n_groups:
    #                 title_appendix += str(param_time) + ";"

    #         plt.plot(X, Y, label ="K = " +  str(n_groups) + ", (" + title_appendix + ")")
    # plt.legend()
    # plt.title(dataset)


    # plt.savefig(os.path.join(PATH_FOLDER, "figures", "DB_score", dataset+ ".png"))
    # # plt.show()
    # plt.close()




# # To check with Vincent
# for dataset in datasets:
#     plt.figure()

#     # best_groups_per_time_param = {}
#     # for param_time in timeParams:

#     #     vp_best_group = {"method": "Gamma_MC_C1", "dataset":dataset, "param_time": param_time}
#     #     best_groups_per_time_param[param_time] = best_groups_and_scores[buildBestOfGroupsName(vp_best_group)][0]

#     # print(best_groups_per_time_param)
    
#     # times = [15, 285]
#     # times = [37, 703]
#     times = [28, 532]
#     print(times)

#     for t in times:
#         Y_bouldin = []
#         Y_score = []
#         X =[]
        
#         for n_groups in nbGroups:
#             vp = {"method": "Gamma_MC_C1", "dataset":dataset, "n_groups": n_groups}
#             with open(os.path.join(PATH_FOLDER, buildModelName(vp)+ ".pkl" ), 'rb') as f:
#                 model  = pickle.load(f)
#                 clusterings = model.clusterings

#             # for t in clusterings.keys():
            
#             # print (clusterings.keys())
#             with open(os.path.join(PATH_DATA, dataset, "ep_probas_" + str(t) + ".pkl"), 'rb') as g:
#                 data = pickle.load(g)
#                 Y_bouldin.append(davies_bouldin_score(data, clusterings[t].predict(data)))
            

#             Y_score.append(clusterings[t].score(data))
#             X.append(n_groups)

#         plt.close()
#         plt.figure()
#         plt.plot(X,Y_bouldin, label = 'Davies-Bouldin score')
#         plt.title(dataset + " - " + str(t))
#         plt.legend()
#         plt.savefig(dataset + "- alpha=" + str(t) + ".png")
#         plt.show()


#         plt.close()
#         plt.figure()
#         plt.plot(X,Y_score, label = 'score (1/error)')
#         plt.title(dataset + " - " + str(t))
#         plt.legend()
#         plt.title(dataset)
#         plt.show()

#     # plt.savefig(os.path.join(PATH_FOLDER, "figures", "DB_score", dataset+ ".png"))
#     # plt.show()
#     plt.close()

    

# # Add class to the pickles

# datasets = ["Cricket_X", "NonInvasiveFatalECG_Thorax1", "OliveOil", "OSULeaf"]

# import pandas as pd
# import numpy as np

# for dataset in datasets:

#     with open(os.path.join(PATH_DATA, dataset, "ep_probas.pkl"), 'rb') as g:
#         data = pickle.load(g)

#     data_ep_raw = pd.read_csv(os.path.join(PATH_DATA, dataset, dataset+"_ESTIMATE_PROBAS.tsv"), sep='\t', header=None, index_col=None)

#     # print(np.array(data_ep_raw[0]))
#     # print(data)
#     for t in data.keys():
#         # print(np.column_stack((data[t], np.array(data_ep_raw[0]))))   
#         data[t] = np.column_stack((data[t], np.array(data_ep_raw[0])))
#         print(data[t].shape)
#     with open("..\\..\\..\\..\\rendus\\proba_pour_verif\\" + dataset + "\\ep_probas_classes.pkl", "wb") as h:
#         pickle.dump(data, h)    




# Kmeans rapport (Vincent)


# nb of examples function of nb classes
# X = [datasets_nb_classes[dataset] for dataset in datasets]
# Y = [datasets_ep_shapes[dataset][0] for dataset in datasets]

# plt.scatter(X, Y, s= 20)
# for i, dataset in enumerate(datasets):
#     plt.annotate(dataset, (X[i], Y[i]))
# plt.show()


#

##############################################################################################################
# Plot number examples, number classes, ...
##############################################################################################################


# PATH_DAVIES_BOULDIN = os.path.join(PATH_FIGURES, "davies_bouldin")
# if not os.path.exists(PATH_DAVIES_BOULDIN):
#     os.mkdir(PATH_DAVIES_BOULDIN)

# for param_time in timeParams:
#     X = []
#     Y  = [] 
#     plt.figure(figsize=(12, 12), dpi=240)
#     for i, dataset in enumerate(datasets):
#         vp_best_group = {"method": "Gamma_MC_C1", "dataset":dataset, "param_time": param_time}
        
#         N = datasets_ep_shapes[dataset][0]
#         nb_classes = datasets_nb_classes[dataset]
#         Kopt = best_groups_and_scores[buildBestOfGroupsName(vp_best_group)][0] 
        
#         x= N/nb_classes
#         y = np.abs(Kopt- nb_classes )/nb_classes
#         Y.append(y)
#         X.append(x)
        
#         # plt.annotate(str(dataset)+","+str(nb_classes) +"," +str(Kopt), (x,y))
#         plt.annotate(str(nb_classes) +"," +str(Kopt), (x,y), size = 8)

#     plt.xlabel(r"$N/nb_{classes}$")
#     plt.ylabel(r"$|K_{opt}-nb_{classes}|/nb_{classes}$")
#     plt.scatter(X,Y, s = 16)
#     plt.title("alpha=" + str(param_time))
#     plt.savefig(os.path.join(PATH_DAVIES_BOULDIN, "nb_classes_examples_alpha=" + str(param_time) + ".png"))
#     # plt.show()









##############################################################################################################
# Print table with datasets, number of examples and number of classes and number of groups optimal
##############################################################################################################

# for param_time in timeParams:
#     print("## alpha = " + str(param_time))
#     print("| dataset| N| nb classes|K opt|")
#     print("| ------------- |:-------------:| -----:|-----:|")

#     for i, dataset in enumerate(datasets):
#         vp_best_group = {"method": "Gamma_MC_C1", "dataset":dataset, "param_time": param_time}
#         N = datasets_ep_shapes[dataset][0]
#         nb_class = datasets_nb_classes[dataset]
#         Kopt = best_groups_and_scores[buildBestOfGroupsName(vp_best_group)][0]
#         print("|",dataset,"|", N ,"|", nb_class, "|", Kopt , "|" )
#     print()



# ratios_errors = {}
# for param_time in timeParams:
#     print("## alpha = " + str(param_time))
#     print("| dataset| N/nb classes | (K opt-nb_class)/nb_class|")
#     print("| ------------- |:-------------:|-----:|")

#     for i, dataset in enumerate(datasets):
#         vp_best_group = {"method": "Gamma_MC_C1", "dataset":dataset, "param_time": param_time}
#         N = datasets_ep_shapes[dataset][0]
#         nb_class = datasets_nb_classes[dataset]
#         Kopt = best_groups_and_scores[buildBestOfGroupsName(vp_best_group)][0]
#         # print("|",dataset,"|", N/nb_class ,"|", abs(nb_class - Kopt)/nb_class , "|" )

#         ratios_errors[dataset] = {"ratio": N/nb_class, "error": abs(nb_class - Kopt)/nb_class}
#     print()


# ##############################################################################################
# # Plot matrices
# ##############################################################################################
# PATH_MODELS = "experiments/results tornado/models_explo"

# for param_time in timeParams:
#     X = []
#     Y = []
#     plt.figure()

#     for dataset in datasets:
#         vp_best_group = {"method": "Gamma_MC_C1", "dataset":dataset, "param_time": param_time}
#         Kopt = best_groups_and_scores[buildBestOfGroupsName(vp_best_group)][0] 

#         vp = {"method": "Gamma_MC_C1", "dataset":dataset, "n_groups": Kopt}
#         with open(os.path.join(PATH_MODELS, buildModelName(vp)+ ".pkl" ), 'rb') as f:
#             model  = pickle.load(f)
        
#         count_zero_components = 0
#         for matrix in model.transitionMatrices.values():
#             count_zero_components += np.sum(matrix)

#         vp_best_group = {"method": "Gamma_MC_C1", "dataset":dataset, "param_time": param_time}
        
#         N = datasets_ep_shapes[dataset][0]
#         nb_classes = datasets_nb_classes[dataset]

#         a = np.abs(Kopt- nb_classes )/nb_classes
        
#         x = a/(Kopt**2)
#         y = count_zero_components
#         X.append(x)
#         Y.append(y)

#         plt.annotate(dataset, (x,y), size = 4)

#     plt.scatter(X,Y)
#     plt.xlabel(r"$|K_{opt}-nb_{classes}|/nb_{classes}$")
#     plt.ylabel("nb_zeros/K^2")
#     plt.title(str(param_time))
#     plt.show()
#     # plt.close()
#     # plt.matshow(model.transitionMatrices[t], cmap = 'gray')
#     # plt.title(dataset + " - " + "K = " + str(n_groups) + " - t = " + str(t) )
#     # print(model.transitionMatrices[t])
#     # plt.savefig("matrices/" + dataset + "_" + str(n_groups) + "_" + str(t) + ".png")
#     # # plt.show()




####################################################################################
# CORRELATION Kopt / nb_classes
####################################################################################



plt.figure()
for param_time in timeParams:
    X = []
    Y = []
    for i, dataset in enumerate(datasets):
        vp_best_group = {"method": "Gamma_MC_C1", "dataset":dataset, "param_time": param_time}
        nb_class = datasets_nb_classes[dataset]
        Kopt = best_groups_and_scores[buildBestOfGroupsName(vp_best_group)][0]

        X.append(nb_class)
        Y.append(Kopt)
    plt.scatter(X,Y, s = 10, label = str(param_time))

plt.xlabel("Number of classes")
plt.ylabel("Optimized K")
plt.legend()
plt.show()

