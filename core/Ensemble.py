import pickle as pkl
import numpy as np

from Vectorizer import Vectorizer
from Channel import Channel, ChannelHub
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone

class Ensemble(Exception):
    '''
    Ensemble class wrapper around sklearn classifiers
    '''

    def __init__(self, vectorizerHub):
        '''
        Constructor
        '''

        if not vectorizerHub:
            raise Exception("vectorizerHub reference is needed")

        # private reference to certain named hubs
        self.vectorizerHub = vectorizerHub
        self.clfs = []
        self.isFit = False

    # TODO: set verbosity to False
    def fit(self, keys, channelHub, Y, clf=None, verbose=True):
        '''
        Method to fit data to the classifier
        '''

        ## Initialize the classifier if not present
        if not clf:
            if verbose:
                print ("Ensemble > clf not present, using LogisticRegression() by default")

            clf = LogisticRegression()

        ## clear existing classifiers
        if verbose:
            print ("Ensemble > clearning existing classifiers")

        self.clfs = []

        ## for each name in hub key, create a classifiers
        for name in keys:
            if verbose:
                print ("Ensemble > training for key: %s" % name)
            
            vec = self.vectorizerHub.get(name)
            ## ^ above one should be a trained vectorizer

            X = channelHub.get(name).get()
            X = vec.transform(X)

            _clf = clone(clf)
            _clf.fit(X, Y)
            self.clfs.append({
                'clf': _clf,
                'name': name
            })

        self.isFit = True

    def predict(self, channelHub, threshold=0.5):
        '''
        Method to predict class for data input
        '''

        if not self.isFit:
            raise Exception("Cannot predict before fit")

        y_pred = self.predict_proba(channelHub)
        for i, y in enumerate(y_pred):
            if y >= threshold:
                y_pred[i] = 1
            else:
                y_pred[i] = 0
                
        return y_pred
    
    # TODO: set verbosity to False    
    def predict_proba(self, channelHub, verbose=True):
        if not self.isFit:
            raise Exception("Cannot predict before fit")

        ys = []
        for clf in self.clfs:
            _clf = clf['clf']
            name = clf['name']
            if verbose:
                print ("Ensemble > predict_proba > predicting with %s" % name)

            X = channelHub.get(name).get()
            X = self.vectorizerHub.get(name).transform(X)

            y_pred = _clf.predict_proba(X)
            ys.append(y_pred)

        return self.__merge(np.array(ys))

    def to_disk(self, filename):
        '''
        Method to persist the ensemble to disk
        '''
        if not self.vectorizer:
            raise Exception("Vectorizer > fit before persisting")

        with open(filename, "wb") as ofp:
            pkl.dump(self.clfs, ofp)

        # TODO: set default verbose to false

    @staticmethod
    def from_disk(filename, vectorizerHub, verbose=True):
        '''
        Static method to load the vectorizer from disk and return an instance of
        this class
        '''

        ensemble = None

        if verbose:
            print ("Ensemble > loading data from %s" % filename)

        with open(filename, "rb") as ifp:
            print ("Ensemble > File %s found, creating Ensemble Instance" % filename)            
            clfs = pkl.load(ifp)
            ensemble = Vectorizer(vectorizerHub)
            ensemble.clfs = clfs
            ensemble.isFit = True
        
        return ensemble

    def __merge(self, ys):
        '''
        Private Method:
        Merge results of N classifiers
        ys should be a A * B * 2 shaped array
        A is number of classifiers
        B is number of rows for prediction
        '''

        result = []
        for i in range(ys.shape[1]):
            # sum_0 = 0
            sum_1 = 0
            for y in ys:
                # sum_0 += y[i][0]
                sum_1 += y[i][1]

            # sum_0 /= ys.shape[0]
            sum_1 /= ys.shape[0]
            result.append(sum_1)

        return result