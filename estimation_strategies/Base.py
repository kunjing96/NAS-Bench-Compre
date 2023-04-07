from estimation_strategies import _register


@_register
class Base():

    def __init__(self, config, search_space):
        self.config = config
        self.search_space = search_space

    def __call__(self, arch):
        raise NotImplementedError('Method __call__ of class {} is not implemented.'.format(self.__class__.__name__))
