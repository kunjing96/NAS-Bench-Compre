import os
from lib.config import get_config
from pathlib import Path

from search_spaces import get_search_space, available_search_spaces
from search_strategies import get_search_strategy, available_search_strategies
from estimation_strategies import get_estimation_strategy, available_estimation_strategies


class API():
    """API for NAS-Bench"""

    def __init__(self, config):
        """
        args:
            config: Hyperparameters dict for search space, search strategy, estimation strategy.
        """
        self.config = config
        self.search_space           = get_search_space(config.SEARCHSPACE)
        self.estimation_strategy    = get_estimation_strategy(config.ESTIMATIONSTRATEGY, self.search_space)
        self.search_strategy        = get_search_strategy(config.SEARCHSTRATEGY, self.search_space, self.estimation_strategy)

    def run(self):
        return self.search_strategy()


if __name__ == '__main__':
    print(available_search_spaces())
    print(available_search_strategies())
    print(available_estimation_strategies())
    config = get_config(cfg_file=os.path.join('configs', 'base.yaml'))
    if config.OUTPUT:
        Path(config.OUTPUT).mkdir(parents=True, exist_ok=True)
    print(config)
    api = API(config=config)
    print(api)
    print(api.config)
    print(api.search_space)
    print(api.search_strategy)
    print(api.estimation_strategy)
    print('Search Start')
    optimal, history, time_cost = api.run()
    print(optimal)
    print(history)
    print(time_cost)
    print('Search End')