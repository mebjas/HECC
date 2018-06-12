import numpy as np

from Channel import ChannelHub
from Ensemble import Ensemble

class OneVsRest(Exception):
    def __init__(self):
        self.classifiers = {}

    ## TODO: set verbosity to false
    def add(self, key, ensemble, overwrite=False, verbose=True):
        '''
        Method to add an item to hub
        '''
        if key in self.classifiers and not overwrite:
            raise Exception(
                "KeyEror: %s already exists in OneVsRest, set {overwrite} = True to overwrite" % key)

        self.classifiers[key] = ensemble
        if verbose:
            print ("OneVsRest > %s added to ensemble" % key)

    def get_keys(self):
        return list(self.classifiers.keys())
    
    def predict(self, channelHub, threshold=0):
        labelIndex = {}
        results = []    
        for key, ensemble in self.classifiers.items():
            i = len(self.labelIndex)
            labelIndex[i] = key

            results.append(ensemble.predict_proba(channelHub))

        return self.__merge(np.array(results), labelIndex)

    def __merge(self, results, labelIndex):
        # results would be a N X M X 1 array
        # N ensembles
        # M rows
        mresults = []
        for i in range(results.shape[1]):
            max_proba = 0
            label = None
            for j, result in enumerate(results):
                if result[i] > max_proba:
                    max_proba = result[i]
                    label = labelIndex[j]
            
            mresults.append({
                "label": label,
                "proba": max_proba
            })

        return mresults


