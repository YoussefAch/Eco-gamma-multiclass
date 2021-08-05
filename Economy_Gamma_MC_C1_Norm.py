"""
    Author : Youssef Achenchabe
    Orange labs

    MULTICLASS : GROUPING METHOD = Clustering ( uses a clustering method to form groups in the Economy method)
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
# from utils import generatePossibleSequences, margins, sigmoid, entropyFunc1, giniImpurity
from utils import *
try:
    import cPickle as pickle
except ImportError:
    import pickle
from sklearn.model_selection import StratifiedShuffleSplit
from utils import transform_to_format_fears
import json
from Economy import Economy
import os.path as op
import random
import bisect



class Economy_Gamma_MC_C1_Norm(Economy):

    """
    Economy_Gamma inherits from Economy

    ATTRIBUTES :

        - nbIntervals : number of intervals.
        - order       : order of markov chain.
        - thresholds  : dictionary of thresholds for each time step.
        - transitionMatrices   : transition matrices for each sequence (t,t+1).
        - complexTransitionMatrices : transition matrices for each sequence (t-ordre..t,t+1).
        - indices     : indices of data associated to each time step and each interval
        - labels     : list of labels observed on the data set.
    """

    def __init__(self, misClassificationCost, timeCost, min_t, classifier, nbIntervals, order, sampling_ratio, use_complete_ECO_model, pathECOmodel, fears, dataset , feat, folderRealData):
        super().__init__(misClassificationCost, timeCost, min_t, classifier)
        self.method = "Gamma_MC_C1_Norm"
        self.nbIntervals = nbIntervals
        self.order = order
        self.thresholds = {}
        self.transitionMatrices = {}
        self.complexTransitionMatrices = {}
        self.indices = {}
        self.P_y_gamma = {}
        self.use_complete_ECO_model = use_complete_ECO_model
        self.pathECOmodel = pathECOmodel
        self.sampling_ratio = sampling_ratio
        self.fears = fears
        self.dataset = dataset
        self.feat = feat
        self.folderRealData = folderRealData
        self.ifProba = True
        self.clusterings = {}
        self.calibrations_grids = {}
        self.calibration_ranks = {}
        self.N_calibration = 1000

    def fit(self, train_classifiers, estimate_probas, ratioVal, path, usePretrainedClassifs=True):
        """
           This procedure fits our model Economy

           INPUTS :
                X : Independent variables
                Y : Dependent variable

           OUTPUTS :
                self.thresholds : dictionary of thresholds for each time step.
                self.transitionMatrices   : transition matrices for each sequence (t,t+1).
                self.P_yhat_y : confusion matrices
                self.indices  : indices associated to each time step and each interval
        """

        self.max_t = train_classifiers.shape[1] - 1

        step = int(self.max_t*self.sampling_ratio) if int(self.max_t*self.sampling_ratio)>0 else 1
        self.timestamps = [t for t in range(step, self.max_t + 1, step)]

        if self.use_complete_ECO_model:
            print("USE COMPLETE ECO --> NOT PLANNED !!!!!!")
            with open(self.pathECOmodel, 'rb') as input:
                fullmodel = pickle.load(input)

            for t in self.timestamps:
                self.classifiers[t] = fullmodel.classifiers[t]
            self.labels = fullmodel.labels

            for i,t in enumerate(self.timestamps):

                self.thresholds[t] = fullmodel.thresholds[t]
                self.indices[t] = fullmodel.indices[t]
                # compute simple transition matrices for t > min_t
                if (i>0):
                    self.transitionMatrices[t] = self.computeTransitionMatrix(self.indices[t-step*i], self.indices[t])

                # compute complex transition matrices
                if (t>= self.min_t + (self.order-self.min_t)*(self.order>self.min_t) and self.order != 1):
                    self.complexTransitionMatrices[t] = self.computeComplexTransitionMatrix([self.indices[t-i-1] for i in range(self.order-1)], self.indices[t])

                if (t>=self.min_t):
                    self.P_yhat_y[t] = fullmodel.P_yhat_y[t]
                    self.P_y_gamma[t] = fullmodel.P_y_gamma[t]
        else:

            ## Split into train and val
            Y_train = train_classifiers.iloc[:, 0]
            X_train = train_classifiers.loc[:, train_classifiers.columns != train_classifiers.columns[0]]
            Y_val = estimate_probas.iloc[:, 0]
            X_val = estimate_probas.loc[:, estimate_probas.columns != estimate_probas.columns[0]]

            # labels seen on the data set
            self.labels = Y_train.unique()


            # time step to start train classifiers
            starting_timestep = self.min_t

            # fit classifiers
            #self.fit_classifiers(X_train, Y_train, starting_timestep, usePretrainedClassifs, path)


            # iterate over all time steps 1..max_t
            for i,t in enumerate(self.timestamps):

                #compute thresholds and indices
                self.thresholds[t], self.indices[t] = self.computeClusteringAndIndices(X_val.iloc[:, :t], t)

                if (i>0):
                    self.transitionMatrices[t] = self.computeTransitionMatrix(self.indices[t-step*i], self.indices[t])
                # compute complex transition matrices
                if (t>= self.min_t + (self.order-self.min_t)*(self.order>self.min_t) and self.order != 1):
                    self.complexTransitionMatrices[t] = self.computeComplexTransitionMatrix([self.indices[t-i-1] for i in range(self.order-1)], self.indices[t])

                # compute confusion matricies
                if (t>=self.min_t):
                    self.P_yhat_y[t] = self.compute_P_yhat_y_gammak(X_val, Y_val, t, self.indices[t])

                    self.P_y_gamma[t] = self.compute_P_y_gamma(Y_val, self.indices[t])

            # print(self.indices)



    
    def max(self, proba_vector):
        return np.array(proba_vector).sort()[-1]



    def calibration(self, probas, rounding_power):

        """Build calibration mapping along each axis of probas input
        Parameters
        ----------
        probas : numpy array
            Classifier outputs to build calibration on
        N_max : int
            Max number of samples on the calibration
        
        Returns
            Array of values, array of calibrated ranks
        -------
        
        """
        N,K = probas.shape
        M = 10**rounding_power

        calibrated_values = np.array([i/N for i in range(N)])

        list_grids = []
        
        # If less than M examples, return all values
        if N < M:
            return [np.array([np.sort(probas[:,i]), calibrated_values]).transpose() for i in range(K)] 
        # Else, round values and merge equal ones
        else:
            for i in range(K):
                
                values = []
                ranks = []
                
                sorted_values = np.sort(probas[:,i])
                uniques, counts = np.unique( np.round(sorted_values, decimals=rounding_power) , return_counts = True)
                cumulated_indexes = np.cumsum(counts)
                
                for l in range(len(cumulated_indexes)): 
                    values.append(uniques[l])
                    ranks.append(np.mean(calibrated_values[cumulated_indexes[l-1]%N:cumulated_indexes[l]]))
                list_grids.append(np.array([values, ranks]).transpose())

            return(list_grids)
        

    def calibrate(self, list_grids, probas_input):
    
        """Calibrates the probas_input values according to the calibration_grid and ranks
        Parameters
        ----------
        calibration_grid : list of shape (N,2) numpy arrays that associate values and calibrated ranks
        probas_input: array of probability vectors to calibrate
        Returns
            Numpy array of calibrated values
        -------
        
        """
        
        K = len(list_grids)
        N = probas_input.shape[0]
        
        calibrated_2 = np.zeros((N,K))
        
        for i in range(K):
            M_cal,_ = list_grids[i].shape
            calibration_values = list_grids[i][:,0]
            calibration_ranks = list_grids[i][:,1]
            for j in range(N):
                x =  probas_input[j,i]
                bisected_right = bisect.bisect(calibration_values, x)
                if bisected_right == 0:
                    u = 0
                    a= 0
                else:
                    u = calibration_values[bisected_right-1]
                    a = calibration_ranks[bisected_right -1]
                if bisected_right == M_cal:
                    v = 1
                    b = 1
                else:
                    v = calibration_values[bisected_right]
                    b = calibration_ranks[bisected_right]
                    
                o = a + (b-a)*(x-u)/(v-u)
                
                calibrated_2[j,i] = o
        return calibrated_2


    def calibrate_vector(self, list_grids, proba_vector):
    
        """Calibrates the probas_input values according to the calibration_grid and ranks
        Parameters
        ----------
        calibration_grid : list of shape (N,2) numpy arrays that associate values and calibrated ranks
        probas_input: array of probability vectors to calibrate
        Returns
            Numpy array of calibrated values
        -------
        
        """
        
        K = len(list_grids)
        
        calibrated_2 = np.zeros(K)
        
        for i in range(K):
            M_cal,_ = list_grids[i].shape
            calibration_values = list_grids[i][:,0]
            calibration_ranks = list_grids[i][:,1]

            x =  proba_vector[i]
            bisected_right = bisect.bisect(calibration_values, x)
            if bisected_right == 0:
                u = 0
                a= 0
            else:
                u = calibration_values[bisected_right-1]
                a = calibration_ranks[bisected_right -1]
            if bisected_right == M_cal:
                v = 1
                b = 1
            else:
                v = calibration_values[bisected_right]
                b = calibration_ranks[bisected_right]
                
            o = a + (b-a)*(x-u)/(v-u)
            
            calibrated_2[i] = o
        return calibrated_2


    def computeClusteringAndIndices(self, X_val, t):
        print("Computing grouping clusterings")

        """
           This procedure computes clustering models and indices of data associatied
           to each cluster.

           INPUTS :
                X_val : validation data

           OUTPUTS :
                thresholds : dictionary of thresholds for each time step.
                indices  : indices associated to each time step and each interval
        """


        # print("COMPUTING THRESHOLD")
        _, t = X_val.shape

        # Predict classes
        if self.fears:
            predictions = self.handle_my_classifier(t, transform_to_format_fears(X_val), proba=True)
            predictions = predictions.values
        elif self.feat:
            with open(op.join(self.folderRealData, self.dataset, 'ep_probas_'+str(t)+'.pkl') ,'rb') as inp:
                predictions = pickle.load(inp)
        else:
            predictions = self.classifiers[t].predict_proba(X_val)

        # predictions = [self.aggregateProbaVector(proba_vector) for proba_vector in predictions]
        # print("Ready to scale")
        # self.proba_scaler = StandardScaler().fit(predictions)
        # predictions = self.proba_scaler.transform(predictions)

        # Calibrate along each dimension
        self.calibrations_grids[t] = self.calibration(predictions, 3)
        predictions = self.calibrate(self.calibrations_grids[t], predictions)

        # TODO: Put good parameters in Kmeans 
        print("Ready to cluster proba vectors")
        kmeans_model = KMeans(n_clusters= self.nbIntervals, init='k-means++', n_init=10, max_iter=3000, tol=0.0001).fit(predictions)

        self.clusterings[t] = kmeans_model

        indices  = [[] for i in range(self.nbIntervals)]
        clusters = kmeans_model.labels_
        
        for index, cluster in enumerate(clusters):
            indices[cluster].append(index)
        thresholds = None

        return thresholds,indices




    def compute_P_yhat_y_gammak(self, X_val, Y_val, timestep, indicesData):
        """
           This function computes P_t(ŷ/y,c_k)


           INPUTS :
                X_val, Y_val : valdiation data
                timestep     : timestep reached
                indicesData  : indices of data associated to each interval / timestep
           OUTPUTS :
                probabilities : P_t(ŷ/y,gamma_k)

        """

        occurences = {}

        # initialise probabilities to 0
        probabilities = {(gamma_k, y, y_hat):0 for y in self.labels for y_hat in self.labels for gamma_k in range(self.nbIntervals)}
        

        keysprob = probabilities.keys()
        # print(np.unique(np.array([x[0] for x in keysprob])))
        # print(np.unique(np.array([x[1] for x in keysprob])))
        # print(np.unique(np.array([x[2] for x in keysprob])))


        # Iterate over intervals
        for gamma_k in range(self.nbIntervals):

            indices_gamma_k = indicesData[gamma_k]
            # Subset of Validation set in interval gamma_k
            X_val_ck = X_val.loc[indices_gamma_k,:]

            # Subset of Validation set in interval gamma_k
            if (X_val_ck.shape[0]>0):
                if self.fears:
                    predictions = self.handle_my_classifier(timestep, transform_to_format_fears(X_val_ck.iloc[:, :timestep]))
                elif self.feat:
                    with open(op.join(self.folderRealData, self.dataset, 'ep_preds_'+str(timestep)+'.pkl') ,'rb') as inp:
                        predictions = list(pickle.load(inp))
                        predictions = [predictions[ii] for ii in indices_gamma_k]
                else:
                    predictions = self.classifiers[timestep].predict(X_val_ck.iloc[:, :timestep])

                for y_hat, y in zip(predictions, Y_val.loc[indices_gamma_k]):
                    # frequenceuence
                    probabilities[gamma_k, y, y_hat] += 1
        # normalize
        for gamma_k, y, y_hat in probabilities.keys():
            Y_val_gamma = Y_val.loc[indicesData[gamma_k]]

            # number of observations in gammak knowing y
            sizeCluster_gamma = len(Y_val_gamma[Y_val_gamma==y])

            if (sizeCluster_gamma != 0):
                probabilities[gamma_k, y, y_hat] /= sizeCluster_gamma

        return probabilities

    def compute_P_y_gamma(self, Y_val, indicesData):
        """
           This function computes P_t(y|gamma_k)


           INPUTS :
                X_val, Y_val : valdiation data
                timestep     : timestep reached
                indicesData  : indices of data associated to each interval / timestep
           OUTPUTS :
                probabilities : P_t(ŷ/y,gamma_k)

        """

        # Initialize all probabilities with 0
        probabilities = {(gamma_k, y):0 for y in self.labels for gamma_k in range(self.nbIntervals)}

        for gamma_k,e in enumerate(indicesData):
            for ts in e:
                probabilities[gamma_k, Y_val.iloc[ts]] += 1

            if len(e) != 0:
                for y in self.labels:
                    probabilities[gamma_k, y] /= len(e)
        return probabilities

    def computeComplexTransitionMatrix(self, indices_t_preced, indices_t):

        """
           This function computes a transition matrix between indices_t_preced and
           indices_t. (N^delta x N) matrix


           INPUTS :
                indices_t_preced : indices of data individuals in each interval at
                                   time steps considered in the past
                indices_t        : indices of data individuals in each interval at
                                   time step t

           OUTPUTS :
                transMatrix : transition matrix

        """
        transMatrix = np.zeros((self.nbIntervals**self.order, self.nbIntervals))

        # generate possible sequences to have the same order (useful for
        # transition matrix 1..t = t+1)
        possibleSequences = generatePossibleSequences(self.nbIntervals, self.order)

        for i in range(self.nbIntervals**self.order):
            possibleSequence = possibleSequences[:,i]
            # compute the ratio of the time series in gamma_j that were in gamma_i
            for j in range(self.nbIntervals):
                proportion = 0
                for e in indices_t[j]:
                    cond = [e in indices_t_preced[ord][possibleSequence[ord]] for ord in range(self.order-1)]
                    #if all true
                    if sum(cond)==self.order-1:
                        proportion += 1
                size = [len(indices_t_preced[ord][possibleSequence[ord]]) for ord in range(self.order-1)]
                transMatrix[i][j] = proportion/sum(size)
        return transMatrix

    def computeTransitionMatrix(self, indices_t_preced, indices_t):
        """
           This function computes a transition matrix between indices_t_preced and
           indices_t.


           INPUTS :
                indices_t_preced : valdiation data
                indices_t        : timestep reached

           OUTPUTS :
                transMatrix : prior probabilities of label y given a cluster ck.

        """
        # transMatrix = np.zeros((self.nbIntervals, self.nbIntervals))
        # for i in range(self.nbIntervals):
        #     for j in range(self.nbIntervals):
        #         proportion = 0
        #         for e in indices_t[j]:
        #             if e in indices_t_preced[i]:
        #                 proportion += 1
        #         if (len(indices_t_preced[i]) > 0):
        #             transMatrix[i][j] = proportion/len(indices_t_preced[i])
        # return transMatrix

        # matrix = np.array([[ len(set(indices_t[j]).intersection(indices_t_preced[i])) for j in range(self.nbIntervals)] for i in range(self.nbIntervals)]) /  np.array([max(1, len(indices)) for indices in indices_t_preced)]
        return np.array([[ len(set(ind_t).intersection(ind_t_p)) for ind_t in indices_t] for ind_t_p in indices_t_preced]) /  np.array([max(1, len(indices)) for indices in indices_t_preced])[:,None]

    def forecastExpectedCost(self, x_t, pb):
        """
           This function computes expected cost for future time steps given
           a time series xt


           INPUTS :
                x_t : time series

           OUTPUTS :
                totalCosts : list of (max_t - t) values that contains total cost
                             for future time steps.
        """
        t_current = len(x_t)


        # we initialize total costs with time cost
        forecastedCosts = [self.timeCost[t] for t in self.timestamps[self.timestamps.index(t_current):]]
        cost_cm = []
        cost_cd = []
        cost_time = [self.timeCost[t] for t in self.timestamps[self.timestamps.index(t_current):]]
        
        send_alert = True
        p_gamma = np.zeros((1,self.nbIntervals**self.order))

        # Orders > 1 have been removed
        # compute p
        if self.feat:
            interval = self.findInterval(x_t, pb)
        else:
            interval = self.findInterval(x_t)
        # p_gamma[:, interval-1] = 1
        # print("interval", interval)
        # p_gamma[:, interval] = 1
        p_gamma[:, interval ] = 1
        # print("pgamma", p_gamma)

        # we just take into consideration the current instant
        transitionMatrix = p_gamma
        for i,t in enumerate(self.timestamps[self.timestamps.index(t_current):]):

            # Compute p_transition
            if (t>t_current):
                transitionMatrix = np.matmul(transitionMatrix, self.transitionMatrices[t])
            #iterate over intervals
            for gamma_k in range(self.nbIntervals):
                #iterate over possible labels
                for y in self.labels:
                    P_y_gamma = self.P_y_gamma[t]
                    # iterate over possible predictions
                    for y_hat in self.labels:
                        tem = transitionMatrix[:, gamma_k] * self.P_yhat_y[t][gamma_k, y, y_hat] * P_y_gamma[gamma_k, y] * self.misClassificationCost[y_hat-1][y-1]
                        forecastedCosts[i] += tem[0]
            if (i>0):
                if (forecastedCosts[i] < forecastedCosts[0]):
                    send_alert = False
                    break
            cost_cm.append(forecastedCosts[i]-self.timeCost[t])
            cost_cd.append(0)
        return send_alert, [cost_cm, cost_time, cost_cd], forecastedCosts[0]

    def findInterval(self, x_t, pb=None):
        """
           This function finds interval associated with a timeseries given its
           probability

           INPUTS :
                proba : probability given by the classifier at time step t

           OUTPUTS :
                cluster of x_t
        """
        # we could use binary search for better perf
        t_current = len(x_t)
        # predict probability
        if self.fears:
            probadf = self.handle_my_classifier(t_current, transform_to_format_fears(numpy_to_df(x_t)), proba=True)
            proba = probadf['ProbNone1'].values[0]
        elif self.feat:
            proba=pb
        else:
            probadf = self.classifiers[t_current].predict_proba(x_t.reshape(1, -1))
            proba = probadf[0][1] # a verifier

        # Apply calibration on the single vector proba
        proba = self.calibrate_vector(self.calibrations_grids[t_current], proba)
        # Find interval (here: find the cluster)
        ret =  self.clusterings[t_current].predict(np.array([proba]))
        return ret




    def findIntervals(self, x_t):
        """
           This function finds intervals associated with a timeseries given its
           probabilities for last order values


           INPUTS :
                proba : probability given by the classifier at time step t

           OUTPUTS :
                interval of x_t
        """
        intervals = []
        t = len(x_t)
        for i in range(self.order):
            intervals.append(self.findInterval(x_t[:t-i]))
        intervals.reverse()
        return intervals

    # def predict_revocable(self, X_test, oneIDV=None, donnes=None, variante='A'):
        """
        This function predicts for every time series in X_test the optimal time
        to make the prediction and the associated label.

        INPUTS :
            - X_test : Independent variables to test the model

        OUTPUTS :
            - predictions : list that contains [label, tau*] for every
                            time series in X_test.

        """
        if not oneIDV:
            nb_observations, _= X_test.shape
        predictions = []
        if donnes != None:
            test_probas, test_preds = donnes

        # We predict for every time series [label, tau*]
        decisions = []

        

        if oneIDV:
            for t in self.timestamps:
                # first t values of x
                #x = np.array(list(X_test.iloc[i, :t]))
                x = X_test.values[:t]
                
                # compute cost of future timesteps (max_t - t)
                if variante=='A':
                    if not decisions:
                        probass = test_probas[t]
                        proba = probass
                        send_alert, cost = self.forecastExpectedCost(x,proba)
                        if send_alert:
                            decisions.append((t,test_preds[t]))
                    else:
                        if decisions[-1][1] != test_preds[t]:
                            decisions.append((t,test_preds[t]))
                elif variante=='B':
                    probass = test_probas[t]
                    proba = probass
                    send_alert, cost = self.forecastExpectedCost(x,proba)

                    if not decisions:
                        if send_alert:
                            decisions.append((t,test_preds[t]))
                    else:
                        if decisions[-1][1] != test_preds[t] and send_alert:
                            decisions.append((t,test_preds[t]))
                else:
                    probass = test_probas[t]
                    proba = probass
                    send_alert, costs, cost = self.forecastExpectedCost(x,proba)

                    if not decisions:
                        if send_alert:
                            decisions.append((t,test_preds[t]))
                            lastcost = cost
                    else:
                        if decisions[-1][1] != test_preds[t] and send_alert and cost < lastcost:
                            decisions.append((t,test_preds[t]))
        else:
            decisions = []
            cost_estimation = []
            for i in range(nb_observations):
                cost_individual = {}
                dec = []
                rev_cost = []
                # The approach is non-moyopic, for each time step we predict the optimal
                # time to make the prediction in the future.
                for t in self.timestamps:
                    # first t values of x
                    #x = np.array(list(X_test.iloc[i, :t]))
                    x = X_test.iloc[i, :t].values

                    # compute cost of future timesteps (max_t - t)

                    if variante=='A':
                        if not dec:
                            probass = test_probas[t]
                            proba = probass[i]
                            send_alert, cost, c = self.forecastExpectedCost(x,proba)
                            if send_alert:
                                dec.append((t,test_preds[t][i]))
                        else:
                            if dec[-1][1] != test_preds[t][i]:
                                dec.append((t,test_preds[t][i]))
                    elif variante=='B':
                        
                        probass = test_probas[t]
                        proba = probass[i]
                        send_alert, cost, c = self.forecastExpectedCost(x,proba)

                        if not dec:
                            if send_alert:
                                dec.append((t,test_preds[t][i]))
                        else:
                            if dec[-1][1] != test_preds[t][i] and send_alert:
                                dec.append((t,test_preds[t][i]))
                    else:

                        probass = test_probas[t]
                        proba = probass[i]
                        send_alert, costs, cost = self.forecastExpectedCost(x,proba)

                        if not dec:
                            if send_alert:
                                dec.append((t,test_preds[t][i]))
                                lastcost = cost
                        else:
                            if dec[-1][1] != test_preds[t][i] and send_alert and cost < lastcost:
                                dec.append((t,test_preds[t][i]))


             

                decisions.append(dec)
        return decisions