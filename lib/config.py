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
_C.SEARCHSTRATEGY.N = 1000 # Random | AgingEvolution | PredictorBasedRandom default: 1000
_C.SEARCHSTRATEGY.MATUTEPROB = 1.0 # AgingEvolution
_C.SEARCHSTRATEGY.PATIENCEFACTOR = 5 # AgingEvolution | PredictorBasedRandom
_C.SEARCHSTRATEGY.TOURNAMENTSIZE = 10 # AgingEvolution default: 10
_C.SEARCHSTRATEGY.NUMPARENTS = 1 # AgingEvolution
_C.SEARCHSTRATEGY.NUMUTATES = 1 # AgingEvolution default: 1 or 100
_C.SEARCHSTRATEGY.NUMPOPULATION = 64 # AgingEvolution default: 64
_C.SEARCHSTRATEGY.DEVICE = 'cuda' # DARTS default: cuda
_C.SEARCHSTRATEGY.DATASET = '' # DARTS
_C.SEARCHSTRATEGY.DATAPATH = '' # DARTS
_C.SEARCHSTRATEGY.INPUTSIZE = 28 # DARTS
_C.SEARCHSTRATEGY.COLORJITTER = None # DARTS
_C.SEARCHSTRATEGY.AA = None # DARTS
_C.SEARCHSTRATEGY.TRAIN_INTERPOLATION = None # DARTS
_C.SEARCHSTRATEGY.REPROB = 0. # DARTS
_C.SEARCHSTRATEGY.REMODE = None # DARTS
_C.SEARCHSTRATEGY.RECOUNT = None # DARTS
_C.SEARCHSTRATEGY.INATCATEGORY = None # DARTS
_C.SEARCHSTRATEGY.CUTOUT = 0 # AutoFormer | OFA | DARTS
_C.SEARCHSTRATEGY.BATCHSIZE = 64 # DARTS default: 64
_C.SEARCHSTRATEGY.NUMWORKERS = 10 # DARTS default: 10
_C.SEARCHSTRATEGY.PINMEM = True # DARTS default: True
_C.SEARCHSTRATEGY.WEIGHTLR = 0.025 # DARTS
_C.SEARCHSTRATEGY.WEIGHTMOMENTUM = 0.9 # DARTS
_C.SEARCHSTRATEGY.WEIGHTWEIGHTDECAY = 3e-4 # DARTS
_C.SEARCHSTRATEGY.WEIGHTGRADCLIP = 5. # DARTS
_C.SEARCHSTRATEGY.ALPHALR = 3e-4 # DARTS
_C.SEARCHSTRATEGY.ALPHABETA1 = 0.5 # DARTS
_C.SEARCHSTRATEGY.ALPHABETA2 = 0.999 # DARTS
_C.SEARCHSTRATEGY.ALPHAWEIGHTDECAY = 1e-3 # DARTS
_C.SEARCHSTRATEGY.EPOCHS = 50 # DARTS | PredictorBasedRandom
_C.SEARCHSTRATEGY.MINWEIGHTLR = 0.001 # DARTS
_C.SEARCHSTRATEGY.NUMLAYERS = 12 # PredictorBasedRandom
_C.SEARCHSTRATEGY.MODELDIM = 32 # PredictorBasedRandom
_C.SEARCHSTRATEGY.ACT = 'nn.PReLU' # PredictorBasedRandom
_C.SEARCHSTRATEGY.BASEMODEL = 'GATConv' # PredictorBasedRandom
_C.SEARCHSTRATEGY.GRAPHENCDROPOUT = 0.0 # PredictorBasedRandom
_C.SEARCHSTRATEGY.DROPOUT = 0.5 # PredictorBasedRandom
_C.SEARCHSTRATEGY.EDGEDROP1 = 0 # PredictorBasedRandom
_C.SEARCHSTRATEGY.EDGEDROP2 = 0 # PredictorBasedRandom
_C.SEARCHSTRATEGY.FEATDROP1 = 0 # PredictorBasedRandom
_C.SEARCHSTRATEGY.FEATDROP2 = 0 # PredictorBasedRandom
_C.SEARCHSTRATEGY.PROJLAYERS = 64 # PredictorBasedRandom
_C.SEARCHSTRATEGY.FIXEDENCODER = False # PredictorBasedRandom
_C.SEARCHSTRATEGY.LR = 0.001 # PredictorBasedRandom
_C.SEARCHSTRATEGY.ENCODERLR = 0.001 # PredictorBasedRandom
_C.SEARCHSTRATEGY.WEIGHTDECAY = 0.01 # PredictorBasedRandom
_C.SEARCHSTRATEGY.EXPWEIGHTED = False # PredictorBasedRandom
_C.SEARCHSTRATEGY.OPTMODE = 'D' # PredictorBasedRandom
_C.SEARCHSTRATEGY.OPTRESERVEP = 1.0 # PredictorBasedRandom
_C.SEARCHSTRATEGY.NUMINITARCHS = 20 # PredictorBasedRandom
_C.SEARCHSTRATEGY.NUMCANDS = 100 # PredictorBasedRandom
_C.SEARCHSTRATEGY.K = 10 # PredictorBasedRandom
_C.SEARCHSTRATEGY.ENCODERSTATEDICT = None # PredictorBasedRandom
_C.SEARCHSTRATEGY.PREDICTORSTATEDICT = None # PredictorBasedRandom
_C.SEARCHSTRATEGY.PREDICTOR = 'GMAEPredictor' # PredictorBasedRandom

# Estimation strategy config files
_C.ESTIMATIONSTRATEGY = CN()
_C.ESTIMATIONSTRATEGY.NAME = ''
_C.ESTIMATIONSTRATEGY.BATCHSIZE = 64 # AutoFormer | OFA | DARTS | StandardTraining default: 64
_C.ESTIMATIONSTRATEGY.MODELPATH = '' # AutoFormer | OFA
    # model_ckpts/AutoFormer/supernet-tiny.pth
    # model_ckpts/AutoFormer/supernet-small.pth
    # model_ckpts/AutoFormer/supernet-base.pth
    # model_ckpts/OFA/ofa_mbv3_d234_e346_k357_w1.0
    # model_ckpts/OFA/ofa_mbv3_d234_e346_k357_w1.2
    # model_ckpts/OFA/ofa_proxyless_d234_e346_k357_w1.3
    # model_ckpts/OFA/ofa_resnet50_d=0+1+2_e=0.2+0.25+0.35_w=0.65+0.8+1.0
_C.ESTIMATIONSTRATEGY.NUMWORKERS = 10 # AutoFormer | OFA | DARTS | StandardTraining default: 10
_C.ESTIMATIONSTRATEGY.PINMEM = True # AutoFormer | OFA | DARTS | StandardTraining default: True
_C.ESTIMATIONSTRATEGY.AMP = True # AutoFormer | OFA default: True
_C.ESTIMATIONSTRATEGY.DATASET = '' # AutoFormer | OFA | DARTS | StandardTraining
_C.ESTIMATIONSTRATEGY.DATAPATH = '' # AutoFormer | OFA | DARTS | StandardTraining
_C.ESTIMATIONSTRATEGY.INPUTSIZE = 224 # AutoFormer | OFA | DARTS | StandardTraining
_C.ESTIMATIONSTRATEGY.COLORJITTER = 0.4 # AutoFormer | OFA | DARTS | StandardTraining
_C.ESTIMATIONSTRATEGY.AA = 'rand-m9-mstd0.5-inc1' # AutoFormer | OFA | DARTS | StandardTraining
_C.ESTIMATIONSTRATEGY.TRAIN_INTERPOLATION = 'bicubic' # AutoFormer | OFA | DARTS | StandardTraining
_C.ESTIMATIONSTRATEGY.REPROB = 0.25 # AutoFormer | OFA | DARTS | StandardTraining
_C.ESTIMATIONSTRATEGY.REMODE = 'pixel' # AutoFormer | OFA | DARTS | StandardTraining
_C.ESTIMATIONSTRATEGY.RECOUNT = 1 # AutoFormer | OFA | DARTS | StandardTraining
_C.ESTIMATIONSTRATEGY.INATCATEGORY = None # AutoFormer | OFA | DARTS | StandardTraining
_C.ESTIMATIONSTRATEGY.NUMCLASSES = None # AutoFormer | OFA | DARTS | StandardTraining
_C.ESTIMATIONSTRATEGY.CUTOUT = 0 # AutoFormer | OFA | DARTS | StandardTraining
_C.ESTIMATIONSTRATEGY.DEVICE = 'cuda' # AutoFormer | OFA | DARTS | StandardTraining default: cuda
_C.ESTIMATIONSTRATEGY.GPU = None # AutoFormer
_C.ESTIMATIONSTRATEGY.DISTRIBUTED = None # AutoFormer
_C.ESTIMATIONSTRATEGY.DISTEVAL = None # AutoFormer
_C.ESTIMATIONSTRATEGY.AUXWEIGHT = 0.4 # DARTS | StandardTraining
_C.ESTIMATIONSTRATEGY.LR = 0.025 # DARTS | StandardTraining
_C.ESTIMATIONSTRATEGY.MOMENTUM = 0.9 # DARTS | StandardTraining
_C.ESTIMATIONSTRATEGY.WEIGHTDECAY = 3e-4 # DARTS | StandardTraining
_C.ESTIMATIONSTRATEGY.EPOCHS = 600 # DARTS | StandardTraining
_C.ESTIMATIONSTRATEGY.DROPPATHPROB = 0.2 # DARTS | StandardTraining
_C.ESTIMATIONSTRATEGY.GRAD_CLIP = 5. # DARTS | StandardTraining
_C.ESTIMATIONSTRATEGY.EVALMODE = '' # DARTS
_C.ESTIMATIONSTRATEGY.TRAINPORTION = 0.9 # StandardTraining

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
