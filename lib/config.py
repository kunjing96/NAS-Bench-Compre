import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# Search space config files
_C.SEARCHSPACE = CN()
_C.SEARCHSPACE.NAME = ''
_C.SEARCHSPACE.MINPARAMS = None # AutoFormerTiny/Small/Base | OFA_MBV3/MBV3L/Proxylexx/ResNet50
_C.SEARCHSPACE.MAXPARAMS = None # AutoFormerTiny/Small/Base | OFA_MBV3/MBV3L/Proxylexx/ResNet50
_C.SEARCHSPACE.MINFLOPS = None # AutoFormerTiny/Small/Base | OFA_MBV3/MBV3L/Proxylexx/ResNet50
_C.SEARCHSPACE.MAXFLOPS = None # AutoFormerTiny/Small/Base | OFA_MBV3/MBV3L/Proxylexx/ResNet50
_C.SEARCHSPACE.MINDELAY = None # AutoFormerTiny/Small/Base | OFA_MBV3/MBV3L/Proxylexx/ResNet50
_C.SEARCHSPACE.MAXDELAY = None # AutoFormerTiny/Small/Base | OFA_MBV3/MBV3L/Proxylexx/ResNet50

# Search strategy config files
_C.SEARCHSTRATEGY = CN()
_C.SEARCHSTRATEGY.NAME = ''
_C.SEARCHSTRATEGY.N = 1000 # Random | AgingEvolution default: 1000
_C.SEARCHSTRATEGY.MATUTEPROB = 1.0 # AgingEvolution
_C.SEARCHSTRATEGY.PATIENCEFACTOR = 5 # AgingEvolution
_C.SEARCHSTRATEGY.TOURNAMENTSIZE = 10 # AgingEvolution default: 10
_C.SEARCHSTRATEGY.NUMPARENTS = 1 # AgingEvolution
_C.SEARCHSTRATEGY.NUMUTATES = 1 # AgingEvolution default: 1 or 100
_C.SEARCHSTRATEGY.NUMPOPULATION = 64 # AgingEvolution default: 64

# Estimation strategy config files
_C.ESTIMATIONSTRATEGY = CN()
_C.ESTIMATIONSTRATEGY.NAME = ''
_C.ESTIMATIONSTRATEGY.BATCHSIZE = 64 # AutoFormer | OFA default: 64
_C.ESTIMATIONSTRATEGY.MODELPATH = '' # AutoFormer | OFA
    # model_ckpts/AutoFormer/supernet-tiny.pth
    # model_ckpts/AutoFormer/supernet-small.pth
    # model_ckpts/AutoFormer/supernet-base.pth
    # model_ckpts/OFA/ofa_mbv3_d234_e346_k357_w1.0
    # model_ckpts/OFA/ofa_mbv3_d234_e346_k357_w1.2
    # model_ckpts/OFA/ofa_proxyless_d234_e346_k357_w1.3
    # model_ckpts/OFA/ofa_resnet50_d=0+1+2_e=0.2+0.25+0.35_w=0.65+0.8+1.0
_C.ESTIMATIONSTRATEGY.NUMWORKERS = 10 # AutoFormer | OFA default: 10
_C.ESTIMATIONSTRATEGY.PINMEM = True # AutoFormer | OFA default: True
_C.ESTIMATIONSTRATEGY.AMP = True # AutoFormer | OFA default: True
_C.ESTIMATIONSTRATEGY.DATASET = '' # AutoFormer | OFA
_C.ESTIMATIONSTRATEGY.DATAPATH = '' # AutoFormer | OFA
_C.ESTIMATIONSTRATEGY.INPUTSIZE = 224 # AutoFormer | OFA
_C.ESTIMATIONSTRATEGY.COLORJITTER = 0.4 # AutoFormer | OFA
_C.ESTIMATIONSTRATEGY.AA = 'rand-m9-mstd0.5-inc1' # AutoFormer | OFA
_C.ESTIMATIONSTRATEGY.TRAIN_INTERPOLATION = 'bicubic' # AutoFormer | OFA
_C.ESTIMATIONSTRATEGY.REPROB = 0.25 # AutoFormer | OFA
_C.ESTIMATIONSTRATEGY.REMODE = 'pixel' # AutoFormer | OFA
_C.ESTIMATIONSTRATEGY.RECOUNT = 1 # AutoFormer | OFA
_C.ESTIMATIONSTRATEGY.INATCATEGORY = None # AutoFormer | OFA
_C.ESTIMATIONSTRATEGY.NUMCLASSES = None # AutoFormer | OFA
_C.ESTIMATIONSTRATEGY.DEVICE = 'cuda' # AutoFormer | OFA default: cuda
_C.ESTIMATIONSTRATEGY.GPU = None # AutoFormer
_C.ESTIMATIONSTRATEGY.DISTRIBUTED = None # AutoFormer
_C.ESTIMATIONSTRATEGY.DISTEVAL = None # AutoFormer

_C.OUTPUT = ''
_C.TAG = ''


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, cfg_file):
    _update_config_from_file(config, cfg_file)
    config.defrost()
    config.OUTPUT = os.path.join(config.OUTPUT, config.SEARCHSPACE.NAME+'_'+config.SEARCHSTRATEGY.NAME+'_'+config.ESTIMATIONSTRATEGY.NAME, config.TAG)
    config.freeze()


def get_config(cfg_file):
    config = _C.clone()
    update_config(config, cfg_file)
    return config
