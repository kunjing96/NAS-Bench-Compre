import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# Search space config files
_C.SEARCHSPACE = CN()
_C.SEARCHSPACE.NAME = ''

# Search strategy config files
_C.SEARCHSTRATEGY = CN()
_C.SEARCHSTRATEGY.NAME = ''

# Estimation strategy config files
_C.ESTIMATIONSTRATEGY = CN()
_C.ESTIMATIONSTRATEGY.NAME = ''

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
