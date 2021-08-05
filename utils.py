import numpy as np
import pandas as pd

def generatePossibleSequences(nbIntervals, order):

    """
       This function return the possible sequences of intervals given order
       and number of intervals

    """
    seqBase = np.arange(nbIntervals)
    final = []
    for i in range(order):
        if i==0:
            # repeat
            final.append(np.array([seqBase for _ in range(nbIntervals**(order-1))]).flatten())
        else:
            #duplicate
            seq = np.array([(nbIntervals**i)*[e] for e in seqBase]).flatten()
            #repeat
            final.append(np.array([seq for _ in range(nbIntervals**(order-i-1))]).flatten())
    return np.array(final)

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

def numpy_to_df(x):
    return pd.DataFrame({i+1:e for i,e in enumerate(x)}, index=range(1))




# Dispersion, entropy, confidence functions

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def entropyFunc1( scores_vector):
    """
        Computes Shannon entropy of given vector
    """
    return - np.array(scores_vector)@np.log(np.array(scores_vector))

def entropyRenyi(scores_vector, alpha):
    """
        Computes the Renyi entropy of given vector with parameter alpha
    """
    # print("Alpha: ", alpha)
    return np.log((np.array(scores_vector)**alpha).sum())/(1-alpha)

def margins( scores_vector):
    """
        Returns the mean of distances to the max score of the vector
    """
    s = np.array(scores_vector)
    np.sort(s)
    return s[-1] - s[-2]

def giniImpurity(scores_vector):
    """
        Returns the Gini impurity index
    """
    return 1 - (np.array(scores_vector)**2).sum()

def variance(scores_vector):
    s = np.array(scores_vector)
    return ((s - np.mean(s))**2).sum()

def custom1(scores_vector, coeff):
    s = np.array(scores_vector)
    return 1 - (np.array(scores_vector)**coeff).sum()
