import os
import glob


__ESTIMATION_STRATEGY_DICT = {}


def available_estimation_strategies():
    return __ESTIMATION_STRATEGY_DICT.keys()


def get_estimation_strategy(config, search_space):
    return __ESTIMATION_STRATEGY_DICT[config.NAME](config, search_space)


def _register(cls):
    __ESTIMATION_STRATEGY_DICT.update({cls.__name__: cls})
    return cls


__import__(name=os.path.basename(os.path.dirname(__file__)), fromlist=[os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(os.path.dirname(__file__), "[!_]*.py"))])
