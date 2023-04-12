import time
import torch
import torch.nn.functional as F
from timm.utils import accuracy
import tqdm
import logging
import math

from search_strategies import _register
from search_strategies.DARTS import DARTS
from lib.MetricLogger import MetricLogger


@_register
class DARTSAER(DARTS):

    def __init__(self, config, search_space, estimation_strategy):
        super(DARTSAER, self).__init__(config, search_space, estimation_strategy)

    def __call__(self):
        init_time = time.time()
        # training loop
        self.history  = []
        for epoch in range(self.config.EPOCHS):
            self.lr_scheduler.step()
            entropy_reg = 0.2 + (-0.2 - 0.2) * (1 + math.cos(math.pi * epoch / self.config.EPOCHS)) / 2

            logging.info("Epoch: {}".format(epoch))
            self.model.print_alphas()

            # training
            train_res = self.train(entropy_reg)
            logging.info("Train results: {}".format(train_res))

            # validation
            score, perf = self.estimation_strategy(self.model, self.valid_loader)

            self.history.append({
                'arch': self.model.genotype(),
                'alpha': {'normal': [F.softmax(x.detach().cpu(), dim=-1) for x in self.model.alpha_normal], 'reduce': [F.softmax(x.detach().cpu(), dim=-1) for x in self.model.alpha_reduce]},
                'score': score,
                'perf': perf,
                'time': time.time()-init_time,
            })

            torch.cuda.empty_cache()

        return max(self.history, key=lambda x: x['score']), self.history, time.time()-init_time
