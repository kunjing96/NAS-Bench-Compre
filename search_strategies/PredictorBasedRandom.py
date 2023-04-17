import time
import copy
import numpy as np
import torch
import tqdm
from torch.distributions.bernoulli import Bernoulli
from itertools import chain
from torch_geometric.data import Data, Batch, DataLoader
import random

from search_strategies import _register
from search_strategies.Base import Base
from lib.predictor import get_predictor
from lib.arch_datasets import Dataset
from lib.loss import BPRLoss, calculate_fisher


@_register
class PredictorBasedRandom(Base):

    def __init__(self, config, search_space, estimation_strategy):
        super(PredictorBasedRandom, self).__init__(config, search_space, estimation_strategy)
        self.predictor = get_predictor(len(search_space.choices)+3, config)
        if config.ENCODERSTATEDICT is not None:
            self.predictor.encoder.load_state_dict(torch.load(config.ENCODERSTATEDICT))
        if config.PREDICTORSTATEDICT is not None:
            self.predictor.predictor.load_state_dict(torch.load(config.PREDICTORSTATEDICT))
        self.predictor = self.predictor.to(config.DEVICE)

    def get_candidates(self, num):
        candidates = []
        for _ in range(self.config.PATIENCEFACTOR):
            for _ in range(num):
                arch = self.search_space.sample()
                if arch not in self.visited:
                    candidates.append(arch)
                    self.visited.append(arch)
                if len(candidates) >= num:
                    return candidates
        return candidates

    def train(self, predictor, history):
        batch_size = (len(history) - 1) // 10 + 1
        # data prepare
        archs = []
        for h in history:
            decoded_arch = self.search_space.decode(h['arch'])
            decoded_arch.update({'acc': h['score']})
            archs.append((decoded_arch, ))
        finetune_dataset = Dataset(search_space=self.search_space, archs=archs)
        finetune_train_loader = DataLoader(finetune_dataset, batch_size=batch_size, shuffle=True)

        # get optimizer
        if self.config.FIXEDENCODER:
            params = [
                {'params': predictor.predictor.parameters(), 'lr': self.config.LR},
            ]
        else:
            params = [
                {'params': predictor.encoder.parameters(), 'lr': self.config.ENCODERLR},
                {'params': predictor.predictor.parameters(), 'lr': self.config.LR},
            ]
        optimizer = torch.optim.AdamW(params, lr=0, weight_decay=self.config.WEIGHTDECAY)

        # get scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.config.EPOCHS, eta_min=1e-4)

        # get criterion
        criterion = BPRLoss(exp_weighted=self.config.EXPWEIGHTED)

        # =================== HACK BEGIN =====================
        if self.config.OPTMODE == 'D':
            gradient_mask = calculate_fisher(finetune_train_loader, predictor, criterion, self.config.OPTRESERVEP)
        else:
            gradient_mask = None
        # =================== HACK END =======================

        for _ in tqdm.tqdm(range(self.config.EPOCHS)):
            scheduler.step()
            if self.config.FIXEDENCODER:
                predictor.encoder.eval()
            else:
                predictor.encoder.train()
            predictor.predictor.train()
            optimizer.zero_grad()
            for batch in finetune_train_loader:
                batch = batch[0]
                x            = batch.x.to(self.config.DEVICE, non_blocking=True)
                edge_index_x = batch.edge_index.to(self.config.DEVICE, non_blocking=True)
                ptr_x        = batch.ptr.to(self.config.DEVICE, non_blocking=True)
                target       = batch.y.to(self.config.DEVICE, non_blocking=True)[ptr_x[:-1]]

                optimizer.zero_grad()

                output = predictor(x, edge_index_x, ptr_x)
                loss = criterion(output.squeeze(), target.squeeze())

                loss.backward()

                # =================== HACK BEGIN =====================
                if self.config.OPTMODE is not None and not self.config.FIXEDENCODER and self.config.OPTRESERVEP < 1:
                    for p in predictor.model.parameters():
                        if p.grad is None:
                            continue
                        if self.config.OPTMODE == 'D' and gradient_mask is not None:
                            if p in gradient_mask:
                                p.grad *= gradient_mask[p] / self.config.OPTRESERVEP
                        else: 
                            # F
                            grad_mask = Bernoulli(p.grad.new_full(size=p.grad.size(), fill_value=self.config.OPTRESERVEP))
                            p.grad *= grad_mask.sample() / self.config.OPTRESERVEP
                # =================== HACK END =======================

                optimizer.step()

        return predictor

    def arch2Data(self, arch):
        available_ops = ['input1', 'input2',] + self.search_space.choices + ['output']
        x = torch.tensor([available_ops.index(x) for x in arch['ops']], dtype=torch.long)
        if 'acc' in arch.keys():
            y = torch.ones_like(x) * arch['acc']
        else:
            y = None
        forward_edges = [[(i, j) for j, x in enumerate(xs) if x > 0] for i, xs in enumerate(arch['adj'])]
        forward_edges = np.array(list(chain(*forward_edges)))
        backward_edges = forward_edges[::-1, ::-1]
        edges = np.concatenate([forward_edges, backward_edges])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        data = Data(x=x, y=y, edge_index=edge_index)
        return data

    @torch.no_grad()
    def predict(self, predictor, candidates):
        predictions = []
        for arch in candidates:
            arch = self.search_space.decode(arch)
            arch = (arch, )
            batch = tuple()
            for i in range(len(arch)):
                _arch = self.arch2Data(arch[i])
                batch = (*batch, Batch.from_data_list([_arch]))
            batch_x = batch[0]
            x            = batch_x.x.to(self.config.DEVICE, non_blocking=True)
            edge_index_x = batch_x.edge_index.to(self.config.DEVICE, non_blocking=True)
            ptr_x        = batch_x.ptr.to(self.config.DEVICE, non_blocking=True)

            output = predictor(x, edge_index_x, ptr_x)

            measure = output.squeeze().detach().cpu().item()

            predictions.append(measure)

        return predictions

    def __call__(self):
        init_time = time.time()
        self.visited = []
        self.history  = []
        while len(self.history) < self.config.NUMINITARCHS:
            arch = self.search_space.sample()
            if arch not in self.visited:
                score, perf = self.estimation_strategy(arch)
                self.visited.append(arch)
                self.history.append({
                    'arch': arch,
                    'score': score,
                    'perf': perf,
                    'time': time.time()-init_time,
                })
        while len(self.history) < self.config.N:
            candidates = self.get_candidates(int(self.config.NUMCANDS))
            predictor = copy.deepcopy(self.predictor)
            finetuned_predictor = self.train(predictor, self.history)
            candidate_predictions = self.predict(finetuned_predictor, candidates)
            candidate_indices = np.argsort(candidate_predictions)
            for i in candidate_indices[-self.config.K:]:
                arch = candidates[i]
                # score = candidate_predictions[i]
                score, perf = self.estimation_strategy(arch)
                self.visited.append(arch)
                self.history.append({
                    'arch': arch,
                    'score': score,
                    'perf': perf,
                    'time': time.time()-init_time,
                })
        return max(self.history, key=lambda x: x['score']), self.history, time.time()-init_time
