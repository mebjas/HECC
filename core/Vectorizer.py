from Channel import Channel, ChannelHub
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time
import pickle as pkl

class Vectorizer(Exception):
    '''
    Class for wrapping sklearn vectorizers
    '''

    def __init__(self, vectorizer=None):
        '''
        Constructor
        '''

        ## TODO: make this private object
        self.vectorizer = vectorizer

    # TODO: set default verbose to false
    def fit(self, channel, stop_words='english', max_df=0.95, max_features=6000, ngram_range=(1,3), verbose=True):
        '''
        Method to fit data to the internal vectorizer
        '''
        
        if self.vectorizer:
            print ("Overwriting the existing vectorizer")

        t0 = time()
        if verbose:
            print ("Vectorizer > starting fit")

        self.vectorizer = TfidfVectorizer(
            stop_words=stop_words,
            max_df=max_df,
            max_features=max_features,
            ngram_range=ngram_range)

        self.vectorizer.fit(channel.get())
        
        if verbose:
            print ("Vectorizer > fit finished [%0.2f s]" % (time() - t0))
            print ("Vectorizer > No of features: %d" % len(self.vectorizer.vocabulary_))

    # TODO: this can be optimized if multiple ensemble use same vectorizer
    def transform(self, X):
        '''
        Method to transform an input data
        '''

        if not self.vectorizer:
            raise Exception("Vectorizer > fit before transformation")

        return self.vectorizer.transform(X)

    def to_disk(self, filename):
        '''
        Method to persist the vectorizer to disk
        '''
        if not self.vectorizer:
            raise Exception("Vectorizer > fit before persisting")

        with open(filename, "wb") as ofp:
            pkl.dump(self.vectorizer, ofp)

    # TODO: set default verbose to false
    @staticmethod
    def from_disk(filename, verbose=True):
        '''
        Static method to load the vectorizer from disk and return an instance of
        this class
        '''

        vec = None

        if verbose:
            print ("Vectorizer > loading data from %s" % filename)

        with open(filename, "rb") as ifp:
            print ("Vectorizer > File %s found, creating Vectorizer Instance" % filename)            
            v = pkl.load(ifp)
            vec = Vectorizer(v)
        
        return vec

class VectorizerHub:
    '''
    Named hub for vectorizers
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.__hub = {}

    ## TODO: set verbosity to false
    def load(self, dataSource, verbose=True):
        '''
        Method to initialize the vectorizers from ChannelHub instance
        '''
        if str(type(dataSource)) == "<class 'Channel.ChannelHub'>":
            for name, channel in dataSource.all().items():
                if verbose:
                    print ("VectorizerHub > loading %s" % name)

                vec = Vectorizer()
                vec.fit(channel)
                self.__hub[name] = vec

    ## TODO: Both VectorizerHub and ChannelHub should inherit from one base class
    def get(self, name):
        '''
        Method to get one item from hub
        '''
        if not name in self.__hub:
            raise Exception("KeyEror: %s not found in VectorizerHub" % name)

        return self.__hub[name]

    def add(self, name, vectorizer, overwrite=False):
        '''
        Method to add an item to hub
        '''
        if name in self.__hub and not overwrite:
            raise Exception("KeyEror: %s already exists in VectorizerHub, set {overwrite} = True to overwrite" % name)

        self.__hub[name] = vectorizer

    def add_from_disk(self, name, filename, overwrite=False):
        if name in self.__hub and not overwrite:
            raise Exception("KeyEror: %s already exists in VectorizerHub, set {overwrite} = True to overwrite" % name)

        vec = Vectorizer.from_disk(filename)
        self.__hub[name] = vec

    def get_keys(self):
        return list(self.__hub.keys())

    def get_segment(self, segment):
        result = []
        for name in segment:
            result.add(self.get(name))

        return result

    def get_full_segment(self):
        return self.get_segment(self.get_keys())

    # TODO: method to persist entire hub to disk
    def to_disk(self):
        pass