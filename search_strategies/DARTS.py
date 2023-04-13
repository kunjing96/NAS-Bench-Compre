import time
import copy
import torch
import torch.nn.functional as F
from timm.utils import accuracy
import logging
from tqdm.contrib import tzip

from search_strategies import _register
from search_strategies.Base import Base
from lib.datasets import build_dataset
from lib.models.DARTS.architect import Architect
from lib.MetricLogger import MetricLogger


@_register
class DARTS(Base):

    def __init__(self, config, search_space, estimation_strategy):
        super(DARTS, self).__init__(config, search_space, estimation_strategy)

        self.device = torch.device(self.config.DEVICE)

        # get data with meta info
        self.train_data, _ = build_dataset(is_train=True, config=self.config)

        # split data to train/validation
        n_train = len(self.train_data)
        split = n_train // 2
        indices = list(range(n_train))
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.config.BATCHSIZE, sampler=train_sampler, num_workers=self.config.NUMWORKERS, pin_memory=self.config.PINMEM)
        self.valid_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.config.BATCHSIZE, sampler=valid_sampler, num_workers=self.config.NUMWORKERS, pin_memory=self.config.PINMEM)

        # model
        self.model = copy.deepcopy(self.search_space.search_model).to(self.device)

        # weights optimizer
        self.weight_optim = torch.optim.SGD(self.model.weights(), self.config.WEIGHTLR, momentum=self.config.WEIGHTMOMENTUM, weight_decay=self.config.WEIGHTWEIGHTDECAY)
        # alphas optimizer
        self.alpha_optim = torch.optim.Adam(self.model.alphas(), self.config.ALPHALR, betas=(self.config.ALPHABETA1, self.config.ALPHABETA2), weight_decay=self.config.ALPHAWEIGHTDECAY)

        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.weight_optim, self.config.EPOCHS, eta_min=self.config.MINWEIGHTLR)
        self.architect = Architect(self.model, self.config.WEIGHTMOMENTUM, self.config.WEIGHTWEIGHTDECAY)

    def train(self, entropy_reg=None):
        metric_logger = MetricLogger(delimiter="  ")
        # header = 'Train:'

        lr = self.lr_scheduler.get_lr()[0]
        self.model.train()

        for (trn_X, trn_y), (val_X, val_y) in tzip(self.train_loader, self.valid_loader):
            trn_X, trn_y = trn_X.to(self.device, non_blocking=True), trn_y.to(self.device, non_blocking=True)
            val_X, val_y = val_X.to(self.device, non_blocking=True), val_y.to(self.device, non_blocking=True)

            # phase 2. architect step (alpha)
            self.alpha_optim.zero_grad()
            self.architect.unrolled_backward(trn_X, trn_y, val_X, val_y, lr, self.weight_optim, entropy_reg=entropy_reg)
            self.alpha_optim.step()

            # phase 1. child network step (w)
            self.weight_optim.zero_grad()
            logits = self.model(trn_X)
            loss = self.model.criterion(logits, trn_y)
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.weights(), self.config.WEIGHTGRADCLIP)
            self.weight_optim.step()

            acc1, acc5 = accuracy(logits, trn_y, topk=(1, 5))

            batch_size = trn_X.size(0)
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
            # metric_logger.meters['loss'].update(loss.item(), n=batch_size)

        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    def __call__(self):
        init_time = time.time()
        # training loop
        self.history  = []
        for epoch in range(self.config.EPOCHS):
            self.lr_scheduler.step()

            logging.info("Epoch: {}".format(epoch))
            self.model.print_alphas()

            # training
            train_res = self.train()
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
