import os
import glob


__PREDICTOR_DICT = {}


def available_predictors():
    return __PREDICTOR_DICT.keys()


def get_predictor(n_vocab, config):
    return __PREDICTOR_DICT[config.PREDICTOR](n_vocab, config)


def _register(cls):
    if cls.__name__ in __PREDICTOR_DICT:
        raise KeyError(f'Duplicated predictor! {cls.__name__}')
    __PREDICTOR_DICT.update({cls.__name__: cls})
    return cls


__import__(name="lib." + os.path.basename(os.path.dirname(__file__)), fromlist=[os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(os.path.dirname(__file__), "[!_]*.py"))])
