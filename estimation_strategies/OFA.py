import os
import torch
import copy
from timm.utils.model import unwrap_model
from timm.utils import accuracy
import logging
import tqdm

from lib.datasets import build_dataset, random_sample_valid_set, build_transform
from lib.MetricLogger import MetricLogger
from lib.download import download_url
from estimation_strategies import _register
from estimation_strategies.Base import Base
from lib.models.OFA.elastic_nn.utils import set_running_statistics


@_register
class OFA(Base):

    def __init__(self, config, search_space):
        super(OFA, self).__init__(config, search_space)

        self.device = torch.device(config.DEVICE)

        dataset_val, _ = build_dataset(is_train=False, config=config, folder_name="train")
        dataset_test, _ = build_dataset(is_train=False, config=config, folder_name="val")

        train_indexes, valid_indexes = random_sample_valid_set(len(dataset_val), 10000)

        sampler_train = torch.utils.data.sampler.SubsetRandomSampler(train_indexes[:2000])
        sampler_val = torch.utils.data.sampler.SubsetRandomSampler(valid_indexes)

        self.train_loader = torch.utils.data.DataLoader(
            dataset_val, batch_size=config.BATCHSIZE,
            sampler=sampler_train, num_workers=config.NUMWORKERS,
            pin_memory=config.PINMEM
        )
        self.val_loader = torch.utils.data.DataLoader(
            dataset_val, batch_size=config.BATCHSIZE,
            sampler=sampler_val, num_workers=config.NUMWORKERS,
            pin_memory=config.PINMEM
        )
        self.test_loader = torch.utils.data.DataLoader(
            dataset_test, batch_size=config.BATCHSIZE,
            shuffle=True, num_workers=config.NUMWORKERS,
            pin_memory=config.PINMEM
        )

        url_base = "https://hanlab.mit.edu/files/OnceForAll/ofa_nets/"
        dir_path, filename = os.path.dirname(config.MODELPATH), os.path.basename(config.MODELPATH)
        self.model = copy.deepcopy(self.search_space.model)
        init = torch.load(download_url(url_base + filename, model_dir=dir_path), map_location="cpu")["state_dict"]
        self.model.load_state_dict(init)

    @torch.no_grad()
    def evaluate(self, data_loader, amp=True, retrain_config=None):
        metric_logger = MetricLogger(delimiter="  ")
        header = 'Test:'

        config = self.search_space.decode(retrain_config)
        model_module = unwrap_model(self.model)
        model_module.set_active_subnet(**config)
        manual_subnet = model_module.get_active_subnet(preserve_weight=True).to(self.device)
        set_running_statistics(manual_subnet, self.train_loader)
        manual_subnet.eval()
        if 'res' in config.keys():
            data_loader.dataset.transform = build_transform(is_train=False, config=self.config, img_size=config['res'])

        # logging.info("sampled model config: {}".format(config))
        # parameters = sum(p.numel() for p in manual_subnet.parameters() if p.requires_grad)
        # logging.info("sampled model parameters: {}".format(parameters))

        # for images, target in metric_logger.log_every(data_loader, 10, header):
        for images, target in tqdm.tqdm(data_loader):
            images = images.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            # compute output
            if amp:
                with torch.cuda.amp.autocast():
                    output = manual_subnet(images)
            else:
                output = manual_subnet(images)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            batch_size = images.shape[0]
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        # logging.info('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
        #     .format(top1=metric_logger.acc1, top5=metric_logger.acc5))

        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    def __call__(self, arch):
        logging.info("Sampled arch: {}".format(arch))
        val_res = self.evaluate(self.val_loader, amp=self.config.AMP, retrain_config=arch)
        test_res = self.evaluate(self.test_loader, amp=self.config.AMP, retrain_config=arch)
        logging.info("Score: {}\tPerformence: {}".format(val_res, test_res))
        return val_res['acc1'], test_res['acc1']
