import pandas

class Channel(Exception):
    '''
    Channel class, has data input
    Can support dataframes and array input
    '''
    def __init__(self, data, colName=None):
        '''Constructor'''
        # define data input for the channel
        self.__data = None
        if str(type(data)) == "<class 'pandas.core.frame.DataFrame'>":
            if not colName:
                raise Exception("{colName} cannot be None Type for dataframe input")

            self.__data = data[colName].values

        elif str(type(data)) == "<class 'list'>":
            self.__data = data
        else:
            raise Exception("Unknown input data type")

    def get(self):
        '''Method to get the data back'''
        return self.__data

class ChannelHub(Exception):
    '''
    Hub for different channels, to be used once in the service
    '''
    def __init__(self):
        self.__hub = {}

    def add(self, name, data, colName=None, overwrite=False):
        if name in self.__hub and not overwrite:
            raise Exception("KeyEror: %s already exists in ChannelHub, set {overwrite} = True to overwrite" % name)

        self.__hub[name] = Channel(data, colName)

    def get(self, name):
        if not name in self.__hub:
            raise Exception("KeyEror: %s not found in ChannelHub" % name)

        return self.__hub[name]

    def all(self):
        return self.__hub
