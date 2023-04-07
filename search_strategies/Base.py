from search_strategies import _register


@_register
class Base():

    def __init__(self, config, search_space, estimation_strategy):
        self.config = config
        self.search_space = search_space
        self.estimation_strategy = estimation_strategy

    def __call__(self):
        raise NotImplementedError('Method __call__ of class {} is not implemented.'.format(self.__class__.__name__))
