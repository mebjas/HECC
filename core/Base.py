## TODO: finish this class, inherit from this
class Hub:
    def __init__(self):
        self.__hub = {}

        def get(self, name):
        '''
        Method to get one item from hub
        '''
        if not name in self.__hub:
            raise Exception("KeyEror: %s not found in VectorizerHub" % name)

        return self.__hub[name]