import os
import pickle
import numpy as np
from TestUtils import buildModelName

datasets = ["Coffee", "Beef", "OliveOil", "Lighting2", "Lighting7", "FaceFour", "ECG200", "Trace", "Gun_Point", "FISH", "OSULeaf", "Synthetic_control", "DiatomSizeReduction", "Haptics", "Cricket_X", "Cricket_Y", "Cricket_Z", "Adiac", "50words", "InlineSkate", "SonyAIBORobotSurface", "SwedishLeaf", "WordsSynonyms", "MedicalImages", "ECGFiveDays", "CBF",
            "SonyAIBORobotSurfaceII", "Symbols", "ItalyPowerDemand", "TwoLeadECG", "MoteStrain", "CinC_ECG_torso", "FaceAll", "NonInvasiveFatalECG_Thorax1", "NonInvasiveFatalECG_Thorax2", "FacesUCR", "MALLAT", "Yoga", "StarLightCurves", "uWaveGestureLibrary_X", "uWaveGestureLibrary_Y", "uWaveGestureLibrary_Z", "ChlorineConcentration", "Two_Patterns", "Wafer"]

PATH_FOLDER = "experiments/results tornado"
PATH_EXPERIMENT = PATH_FOLDER +  "/tor_2_exp1/experiment_tor_2"
PATH_DATA = "RealData-MORI"


nbGroups = range(2, 11)

# Get models
S = []
for dataset in datasets:    
    for n_groups in nbGroups:
        vp = {"method": "Gamma_MC_C1", "dataset":dataset, "n_groups": n_groups}

        with open(os.path.join(PATH_EXPERIMENT, "modelsECONOMY", buildModelName(vp)+ ".pkl" ), 'rb') as f:
            model  = pickle.load(f)
            clusterings = model.clusterings
            
            for t in clusterings.keys():
                # with open(os.path.join(PATH_DATA, dataset, "test_probas_" + str(t) + ".pkl"), 'rb') as g:
                #     data = pickle.load(g)
                #     Y.append(davies_bouldin_score(data, clusterings[t].predict(data)))

                # print(np.sum(clusterings[t].cluster_centers_, axis = 1))
                S.append(np.mean(np.sum(clusterings[t].cluster_centers_, axis = 1)))
    print(dataset)
print(np.mean(np.array(S)))