import random
import copy

from search_spaces import _register
from search_spaces.Base import Base
from models.AutoFormer import Vision_TransformerSuper


@_register
class AutoFormerTiny(Base):

    def __init__(self, config):
        super(AutoFormerTiny, self).__init__(config)
        self.choices = {'num_heads': [3, 4], 'mlp_ratio': [3.5, 4.0], 'embed_dim': [192, 216, 240] , 'depth': [12, 13, 14]}
        self.model = Vision_TransformerSuper(
                        img_size=224,
                        patch_size=16,
                        embed_dim=256, depth=max(self.choices['depth']),
                        num_heads=max(self.choices['num_heads']),mlp_ratio=max(self.choices['mlp_ratio']),
                        qkv_bias=True, drop_rate=0.0,
                        drop_path_rate=0.1,
                        gp=True,
                        num_classes=1000,
                        max_relative_position=14,
                        relative_position=True,
                        change_qkv=True, abs_pos=True
                    )

    def is_valid(self, arch):
        assert isinstance(arch, tuple)
        decoded_arch = self.decode(arch)
        model = copy.deepcopy(self.model)
        if self.config.MAXPARAMS or self.config.MINPARAMS:
            params = model.get_sampled_params_numel(decoded_arch)
            if self.config.MAXPARAMS and params > self.config.MAXPARAMS:
                print('maximum parameters limit exceed')
                return False
            if self.config.MINPARAMS and params < self.config.MINPARAMS:
                print('under minimum parameters limit')
                return False
        if self.config.MAXFLOPS or self.config.MINFLOPS:
            flops = 0 # TODO: get FLOPs
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
        arch = list()
        arch.append(decoded_arch['depth'])
        dimensions = ['mlp_ratio', 'num_heads']
        for dimension in dimensions:
            arch.extend(decoded_arch[dimension])
        arch.append(decoded_arch['embed_dim'][0])
        return tuple(arch)

    def decode(self, arch):
        depth = arch[0]
        decoded_arch = {}
        decoded_arch['depth'] = depth
        decoded_arch['mlp_ratio'] = list(arch[1:depth+1])
        decoded_arch['num_heads'] = list(arch[depth + 1: 2 * depth + 1])
        decoded_arch['embed_dim'] = [arch[-1]]*depth
        return decoded_arch

    def sample(self):
        arch = list()
        dimensions = ['mlp_ratio', 'num_heads']
        depth = random.choice(self.choices['depth'])
        arch.append(depth)
        for dimension in dimensions:
            for _ in range(depth):
                arch.append(random.choice(self.choices[dimension]))
        arch.append(random.choice(self.choices['embed_dim']))
        return tuple(arch)
