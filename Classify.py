import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import balanced_accuracy_score
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import scipy.linalg

class Classify(object):
    def __init__(self, spec, labels, batches, doMT=False):
        self.spec = spec
        self.labels = labels
        self.batches = batches
        self.doMT = doMT     ### if or not do model transfer based on score movement
    
    def model(self, ix_train, ix_test, nPC=10):
        m_PCA = PCA(n_components = nPC).fit(self.spec[ix_train, :])
        
        scr_train = m_PCA.transform(self.spec[ix_train, :])
        scr_test = m_PCA.transform(self.spec[ix_test, :])

        if self.doMT:
            scr_diff = np.mean(scr_train, axis=0) - np.mean(scr_test, axis=0)
            for i in range(scr_train.shape[0]):
                scr_train[i,:] = scr_train[i, :] - scr_diff 
                
        m_LDA = LinearDiscriminantAnalysis().fit(scr_train, self.labels[ix_train])
        
        pred = m_LDA.predict(scr_test)
        
        return pred
    
    def do_CV(self):    ### leave one batch out cross validation
        log = LeaveOneGroupOut()
        self.pred = np.zeros_like(self.labels)
        accs = []
        for i, (train_index, test_index) in enumerate(log.split(self.spec, self.labels, self.batches)):
            self.pred[test_index] = self.model(train_index, test_index)
            accs = np.append(accs, cal_metric(self.labels[test_index], self.pred[test_index]))    
        return accs


def get_meanspec(spec, labels=None):
    
    if labels is None:
        return np.median(spec, 0)
    
    unilabels = np.unique(labels)
    
    meanspec = []
    for l in unilabels:
        meanspec.append(np.median(spec[labels==l,:], 0))

    meanspec = np.row_stack(meanspec)

    return meanspec, unilabels 

def cal_metric(y_true, y_pred, batches=None):
    if batches is None:
        return balanced_accuracy_score(y_true, y_pred)
    else:
        accs = []
        for b in np.unique(batches):
            accs = np.append(accs, balanced_accuracy_score(y_true[batches==b], y_pred[batches==b]))
        return accs
