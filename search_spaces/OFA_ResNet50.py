import random
import copy

from search_spaces import _register
from search_spaces.Base import Base
from models.OFA.model_zoo import ofa_net
from models.OFA.networks import ResNets
from models.OFA.utils.flops_counter import profile


@_register
class OFA_ResNet50(Base):

    def __init__(self, config):
        super(OFA_ResNet50, self).__init__(config)
        self.choices = {'w': [0.65, 0.8, 1.0], 'e': [0.2, 0.25, 0.35], 'd': [0, 1, 2], 'res': [128, 160, 192, 224]}
        self.model = ofa_net('ofa_resnet50', pretrained=False)

    def is_valid(self, arch):
        decoded_arch = self.decode(arch)
        model = copy.deepcopy(self.model)
        model.set_active_subnet(**decoded_arch)
        manual_subnet = model.get_active_subnet(preserve_weight=True)
        if self.config.MAXPARAMS or self.config.MINPARAMS:
            params = sum(p.numel() for p in manual_subnet.parameters() if p.requires_grad)
            if self.config.MAXPARAMS and params > self.config.MAXPARAMS:
                print('maximum parameters limit exceed')
                return False
            if self.config.MINPARAMS and params < self.config.MINPARAMS:
                print('under minimum parameters limit')
                return False
        if self.config.MAXFLOPS or self.config.MINFLOPS:
            flops, _ = profile(manual_subnet, (1, 3, decoded_arch['res'], decoded_arch['res']))
            if self.config.MAXFLOPS and flops > self.config.MAXFLOPS:
                print('maximum flops limit exceed')
                return False
            if self.config.MINFLOPS and flops < self.config.MINFLOPS:
                print('under minimum flops limit')
                return False
        if self.config.MAXDELAY is not None or self.config.MINDELAY is not None:
            delay = 0 # TODO: get Delay
            if self.config.MAXDELAY and delay > self.config.MAXDELAY:
                print('parameters limit exceed')
                return False
            if self.config.MINDELAY and delay < self.config.MINDELAY:
                print('under minimum parameters limit')
                return False
        return True

    def encode(self, decoded_arch):
        return copy.deepcopy(decoded_arch)

    def decode(self, arch):
        return copy.deepcopy(arch)

    def sample(self):
        arch = dict()
        arch['e'] = []
        for block in self.model.blocks:
            arch['e'].append(random.choice(block.expand_ratio_list))
        arch['d'] = [random.choice([max(self.choices['d']), min(self.choices['d'])])]
        for _ in range(len(ResNets.BASE_DEPTH_LIST)):
            arch['d'].append(random.choice(self.choices['d']))
        arch['w'] = [
            random.choice(list(range(len(self.model.input_stem[0].out_channel_list)))),
            random.choice(list(range(len(self.model.input_stem[2].out_channel_list)))),
        ]
        for _, block_idx in enumerate(self.model.grouped_block_index):
            stage_first_block = self.model.blocks[block_idx[0]]
            arch['w'].append(
                random.choice(list(range(len(stage_first_block.out_channel_list))))
            )
        arch['res'] = random.choice(self.choices['res'])
        return arch
