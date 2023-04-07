from search_spaces import _register


@_register
class Base():

    def __init__(self, config):
        self.config = config

    def is_valid(self, arch):
        return NotImplementedError('Method is_valid of class {} is not implemented.'.format(self.__class__.__name__))

    def encode(self, decoded_arch):
        return NotImplementedError('Method encode of class {} is not implemented.'.format(self.__class__.__name__))

    def decode(self, arch):
        return NotImplementedError('Method decode of class {} is not implemented.'.format(self.__class__.__name__))

    def sample(self):
        raise NotImplementedError('Method sample of class {} is not implemented.'.format(self.__class__.__name__))

    def sample_n(self, n):
        archs = []
        while len(archs) < n:
            arch = self.sample()
            if arch not in archs:
                archs.append(arch)
        return archs
