import torch
import numpy as np
import logging

from search_spaces import _register
from search_spaces.Base import Base
from lib.generators import get_generator
from lib.generators.NAG.utils import graph2arch, ModelSpec, CHOICES
from lib.nets.NAG.model import NASNetworkCIFAR, NASNetworkImageNet
from lib.nets.NAG.utils import count_parameters


@_register
class NAG(Base):

    def __init__(self, config):
        super(NAG, self).__init__(config)
        self.choices = CHOICES
        config.defrost()
        config.NUMPERFS = 3
        config.NUMPARAMS = 3
        config.PERFEMBSIZE = 10
        config.PARAMEMBSIZE = 10
        config.LATENTDIM = 100
        config.MAXSTEP = 10
        config.GEMBSIZE = 250
        config.GHIDDENSIZE = 56
        config.GNUMLAYERS = 3
        config.GDROPOUT = 0.5
        config.GAGGR = 'gsum' # mean/gsum
        config.VOCABSIZE = len(self.choices)
        config.freeze()
        self.generator = get_generator(config)
        if config.STATEDICTPATH:
            self.generator.load_state_dict(torch.load(config.STATEDICTPATH))
        self.generator = self.generator.to(config.DEVICE)
        self.model_cls = NASNetworkImageNet if config.IMAGENET else NASNetworkCIFAR

    def is_valid(self, arch):
        assert isinstance(arch, tuple)
        decoded_arch = self.decode(arch)
        n_classes = 10 if self.model_cls is NASNetworkCIFAR else 1000
        layers = 6 if self.model_cls is NASNetworkCIFAR else 4
        channels = 36 if self.model_cls is NASNetworkCIFAR else 48
        model = self.model_cls(n_classes, layers, channels, 1.0, 1.0, True, 100, decoded_arch)
        if self.config.MAXPARAMS or self.config.MINPARAMS:
            params = count_parameters(model)
            if self.config.MAXPARAMS and params > self.config.MAXPARAMS:
                logging.info('maximum parameters limit exceed')
                return False
            if self.config.MINPARAMS and params < self.config.MINPARAMS:
                logging.info('under minimum parameters limit')
                return False
        if self.config.MAXFLOPS or self.config.MINFLOPS:
            flops = 0 # TODO: get FLOPs
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
        return decoded_arch

    def decode(self, arch):
        return arch

    def sample(self):
        while True:
            n = 64
            z = torch.FloatTensor(np.random.normal(0, 1, (n, 100))).to(self.config.DEVICE)
            gen_perf_labels = torch.LongTensor(np.ones(n) * self.config.PERFLABEL).to(self.config.DEVICE)
            if 'MultiObj' in self.config.GENERATOR:
                gen_param_labels = torch.LongTensor(np.ones(n) * self.config.PARAMLABEL).to(self.config.DEVICE)
            else:
                gen_param_labels = None
            gen_conv_edges, gen_conv_nodes, gen_conv_ns, gen_reduc_edges, gen_reduc_nodes, gen_reduc_ns = self.generator(z, gen_perf_labels, gen_param_labels)
            gen_conv_archs = graph2arch(gen_conv_edges.detach(), gen_conv_nodes.detach(), gen_conv_ns.data)
            gen_reduc_archs = graph2arch(gen_reduc_edges.detach(), gen_reduc_nodes.detach(), gen_reduc_ns.data)
            assert len(gen_conv_archs) == len(gen_reduc_archs) == n
            for gen_conv_arch, gen_reduc_arch in zip(gen_conv_archs, gen_reduc_archs):
                conv_model_spec = ModelSpec(matrix=gen_conv_arch[0], ops=gen_conv_arch[1])
                reduc_model_spec = ModelSpec(matrix=gen_reduc_arch[0], ops=gen_reduc_arch[1])
                if conv_model_spec.valid_spec and reduc_model_spec.valid_spec:
                    arch = (conv_model_spec, reduc_model_spec)
                    return tuple(arch)
