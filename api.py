import os
from lib.config import get_config
from pathlib import Path
import logging
import torch

from search_spaces import get_search_space, available_search_spaces
from search_strategies import get_search_strategy, available_search_strategies
from estimation_strategies import get_estimation_strategy, available_estimation_strategies
from lib.logger import prepare_logger


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
    config = get_config(cfg_file=os.path.join('configs', 'OFA_AutoFormer_ViTSoup.yaml'))
    if config.OUTPUT:
        Path(config.OUTPUT).mkdir(parents=True, exist_ok=True)

    prepare_logger(config=config)

    logging.info('Available search spaces:')
    logging.info(available_search_spaces())
    logging.info('Available search strategies:')
    logging.info(available_search_strategies())
    logging.info('Available estimation strategies:')
    logging.info(available_estimation_strategies())

    api = API(config=config)

    logging.info('='*5 + 'Search Start' + '='*5)
    best, history, time_cost = api.run()
    logging.info('*Best:')
    logging.info(best)
    logging.info('History:')
    logging.info(history)
    logging.info('Time cost:')
    logging.info(time_cost)
    torch.save(
        {
            'best': best,
            'history': history,
            'time_cost': time_cost
        },
        os.path.join(config.OUTPUT, 'checkpoint.pt')
    )
    logging.info('='*5 + 'Search End' + '='*5)