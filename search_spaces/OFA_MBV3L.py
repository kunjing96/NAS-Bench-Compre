from search_spaces import _register
from search_spaces.OFA_MBV3 import OFA_MBV3
from lib.models.OFA.model_zoo import ofa_net


@_register
class OFA_MBV3L(OFA_MBV3):

    def __init__(self, config):
        super(OFA_MBV3, self).__init__(config)
        self.choices = {'ks': [3, 5, 7], 'e': [3, 4, 6], 'd': [2, 3, 4], 'res': [128, 160, 192, 224]}
        self.model = ofa_net('ofa_mbv3_d234_e346_k357_w1.2', pretrained=False)
