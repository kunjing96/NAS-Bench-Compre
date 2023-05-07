import random
import copy
import logging
from yacs.config import CfgNode as CN
from timm.utils.model import unwrap_model

from search_spaces import _register
from search_spaces.Base import Base
from lib.nets.ViTSoup import ViTSoup as ViTSoupModel


@_register
class ViTSoup(Base):

    def __init__(self, config):
        super(ViTSoup, self).__init__(config)
        self.choices = {'num_heads': [[2, 3], [4, 6], [8, 12], [16, 24]], 'mlp_ratio': [3.5, 4.0], 'embed_dim': [84, 96, 108] , 'depth': [[2], [2], [6, 7, 8], [2]], 'window_size': [5, 7]}
        self.model = ViTSoupModel(
                        img_size=224,
                        patch_size=4,
                        in_chans=3,
                        num_classes=1000,
                        embed_dim=max(self.choices['embed_dim']),
                        depths=[max(x) for x in self.choices['depth']],
                        num_heads=[max(x) for x in self.choices['num_heads']],
                        window_size=max(self.choices['window_size']),
                        mlp_ratio=max(self.choices['mlp_ratio']),
                        qkv_bias=True,
                        qk_scale=None,
                        drop_rate=0.0,
                        drop_path_rate=0.2,
                        ape=False,
                        patch_norm=True,
                        use_checkpoint=False,
                        scale=False,
                        shift=False
                    )

    def is_valid(self, arch):
        assert isinstance(arch, tuple)
        decoded_arch = self.decode(arch)
        model = copy.deepcopy(self.model)
        model_module = unwrap_model(model)
        model_module.set_sample_config(decoded_arch)
        if self.config.MAXPARAMS or self.config.MINPARAMS:
            params = model_module.params()
            if self.config.MAXPARAMS and params > self.config.MAXPARAMS:
                logging.info('maximum parameters limit exceed')
                return False
            if self.config.MINPARAMS and params < self.config.MINPARAMS:
                logging.info('under minimum parameters limit')
                return False
        if self.config.MAXFLOPS or self.config.MINFLOPS:
            flops = model_module.flops()
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
        arch = []
        arch.append(decoded_arch['embed_dim'])
        arch.append(tuple(decoded_arch['depth']))
        arch.append(tuple(decoded_arch['num_heads']))
        arch.append(tuple(decoded_arch['window_size']))
        arch.append(tuple(decoded_arch['mlp_ratio']))
        return tuple(arch)

    def decode(self, arch):
        decoded_arch = CN()
        decoded_arch['embed_dim'] = arch[0]
        decoded_arch['depth'] = list(arch[1])
        decoded_arch['num_heads'] = list(arch[2])
        decoded_arch['window_size'] = list(arch[3])
        decoded_arch['mlp_ratio'] = list(arch[4])
        return decoded_arch

    def sample(self):
        config = {}
        config['embed_dim'] = random.choice(self.choices['embed_dim'])
        config['depth'] = [ random.choice(depth) for depth in self.choices['depth'] ]
        config['num_heads'] = []
        config['window_size'] = []
        config['mlp_ratio'] = []
        for i, depth in enumerate(config['depth']):
            for _ in range(depth):
                config['num_heads'].append(random.choice(self.choices['num_heads'][i]))
                config['window_size'].append(random.choice(self.choices['window_size']))
                config['mlp_ratio'].append(random.choice(self.choices['mlp_ratio']))
        return self.encode(config)
