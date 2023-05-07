import os
import glob


__GENERATOR_DICT = {}


def available_generators():
    return __GENERATOR_DICT.keys()


def get_generator(config):
    return __GENERATOR_DICT[config.GENERATOR](config)


def _register(cls):
    if cls.__name__ in __GENERATOR_DICT:
        raise KeyError(f'Duplicated generator! {cls.__name__}')
    __GENERATOR_DICT.update({cls.__name__: cls})
    return cls


__import__(name="lib." + os.path.basename(os.path.dirname(__file__)), fromlist=[os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(os.path.dirname(__file__), "[!_]*.py"))])
__import__(name="lib." + os.path.basename(os.path.dirname(__file__)), fromlist=[os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(os.path.dirname(__file__), "[!_]*[!.py]"))])

