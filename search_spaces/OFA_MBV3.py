import random
import copy
import logging

from search_spaces import _register
from search_spaces.Base import Base
from lib.nets.OFA.model_zoo import ofa_net
from lib.nets.OFA.utils.flops_counter import profile


@_register
class OFA_MBV3(Base):

    def __init__(self, config):
        super(OFA_MBV3, self).__init__(config)
        self.choices = {'ks': [3, 5, 7], 'e': [3, 4, 6], 'd': [2, 3, 4], 'res': [128, 160, 192, 224]}
        self.model = ofa_net('ofa_mbv3_d234_e346_k357_w1.0', pretrained=False)

    def is_valid(self, arch):
        decoded_arch = self.decode(arch)
        model = copy.deepcopy(self.model)
        model.set_active_subnet(**decoded_arch)
        manual_subnet = model.get_active_subnet(preserve_weight=True)
        if self.config.MAXPARAMS or self.config.MINPARAMS:
            params = sum(p.numel() for p in manual_subnet.parameters() if p.requires_grad)
            if self.config.MAXPARAMS and params > self.config.MAXPARAMS:
                logging.info('maximum parameters limit exceed')
                return False
            if self.config.MINPARAMS and params < self.config.MINPARAMS:
                logging.info('under minimum parameters limit')
                return False
        if self.config.MAXFLOPS or self.config.MINFLOPS:
            flops, _ = profile(manual_subnet, (1, 3, decoded_arch['res'], decoded_arch['res']))
            if self.config.MAXFLOPS and flops > self.config.MAXFLOPS:
                logging.info('maximum flops limit exceed')
                return False
            if self.config.MINFLOPS and flops < self.config.MINFLOPS:
                logging.info('under minimum flops limit')
                return False
        if self.config.MAXDELAY is not None or self.config.MINDELAY is not None:
            delay = 0 # TODO: get Delay
            if self.config.MAXDELAY and delay > self.config.MAXDELAY:
                logging.info('parameters limit exceed')
                return False
            if self.config.MINDELAY and delay < self.config.MINDELAY:
                logging.info('under minimum parameters limit')
                return False
        return True

    def encode(self, decoded_arch):
        return copy.deepcopy(decoded_arch)

    def decode(self, arch):
        return copy.deepcopy(arch)

    def sample(self):
        arch = dict()
        for k in ['ks', 'e']:
            v = []
            for _ in range(len(self.model.blocks)-1):
                v.append(random.choice(self.choices[k]))
            arch[k] = v
        arch['d'] = []
        for _ in range(len(self.model.block_group_info)):
            arch['d'].append(random.choice(self.choices['d']))
        arch['res'] = random.choice(self.choices['res'])
        return arch
