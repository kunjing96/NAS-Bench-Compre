import torch
import copy
from timm.utils.model import unwrap_model
from timm.utils import accuracy
import logging
import tqdm

from lib.datasets import build_dataset
from lib.MetricLogger import MetricLogger
from estimation_strategies import _register
from estimation_strategies.Base import Base


@_register
class ViTSoup(Base):

    def __init__(self, config, search_space):
        super(ViTSoup, self).__init__(config, search_space)

        self.device = torch.device(config.DEVICE)

        dataset_val, _ = build_dataset(is_train=False, config=config, folder_name="subImageNet")
        dataset_test, _ = build_dataset(is_train=False, config=config, folder_name="val")

        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

        self.val_loader = torch.utils.data.DataLoader(
            dataset_val, batch_size=int(2 * config.BATCHSIZE),
            sampler=sampler_val, num_workers=config.NUMWORKERS,
            pin_memory=config.PINMEM, drop_last=False
        )
        self.test_loader = torch.utils.data.DataLoader(
            dataset_test, batch_size=int(2 * config.BATCHSIZE),
            sampler=sampler_test, num_workers=config.NUMWORKERS,
            pin_memory=config.PINMEM, drop_last=False
        )

        self.model = copy.deepcopy(self.search_space.model).to(self.device)
        self.model.load_state_dict(torch.load(config.MODELPATH)['model'])

    @torch.no_grad()
    def evaluate(self, data_loader, amp=True, retrain_config=None):
        metric_logger = MetricLogger(delimiter="  ")
        # header = 'Test:'

        self.model.eval()
        config = self.search_space.decode(retrain_config)
        model_module = unwrap_model(self.model)
        model_module.set_sample_config(config=config)

        for images, target in tqdm.tqdm(data_loader):
            images = images.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            # compute output
            if amp:
                with torch.cuda.amp.autocast():
                    output = self.model(images)
            else:
                output = self.model(images)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            batch_size = images.shape[0]
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


    def __call__(self, arch):
        logging.info("Sampled arch: {}".format(arch))
        val_res = self.evaluate(self.val_loader, amp=self.config.AMP, retrain_config=arch)
        test_res = self.evaluate(self.test_loader, amp=self.config.AMP, retrain_config=arch)
        logging.info("Score: {}\tPerformence: {}".format(val_res, test_res))
        return val_res['acc1'], test_res['acc1']
