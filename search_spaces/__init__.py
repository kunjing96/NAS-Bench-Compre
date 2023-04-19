import os
import glob


__SEARCH_SPACE_DICT = {}


def available_search_spaces():
    return __SEARCH_SPACE_DICT.keys()


def get_search_space(config):
    return __SEARCH_SPACE_DICT[config.NAME](config)


def _register(cls):
    if cls.__name__ in __SEARCH_SPACE_DICT:
        raise KeyError(f'Duplicated search space! {cls.__name__}')
    __SEARCH_SPACE_DICT.update({cls.__name__: cls})
    return cls


__import__(name=os.path.basename(os.path.dirname(__file__)), fromlist=[os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(os.path.dirname(__file__), "[!_]*.py"))])
