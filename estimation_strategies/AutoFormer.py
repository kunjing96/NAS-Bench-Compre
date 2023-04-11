import torch
import copy
from timm.utils.model import unwrap_model
from timm.utils import accuracy
import logging
import tqdm

# from lib.ddp import init_distributed_mode, get_rank, get_world_size
from lib.datasets import build_dataset
from lib.MetricLogger import MetricLogger
from estimation_strategies import _register
from estimation_strategies.Base import Base


@_register
class AutoFormer(Base):

    def __init__(self, config, search_space):
        super(AutoFormer, self).__init__(config, search_space)

        # init_distributed_mode(config)
        self.device = torch.device(config.DEVICE)
        # seed = random.randint(0, 65535) + get_rank()
        # torch.manual_seed(seed)
        # np.random.seed(seed)
        # random.seed(seed)
        # torch.backends.cudnn.benchmark = True

        dataset_val, _ = build_dataset(is_train=False, config=config, folder_name="subImageNet")
        dataset_test, _ = build_dataset(is_train=False, config=config, folder_name="val")

        # if config.DISTRIBUTED:
        #     num_tasks = get_world_size()
        #     global_rank = get_rank()
        #     if config.DISTEVAL:
        #         if len(dataset_val) % num_tasks != 0:
        #             logging.warning(
        #                 'Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
        #                 'This will slightly alter validation results as extra duplicate entries are added to achieve '
        #                 'equal num of samples per-process.')
        #         sampler_val = torch.utils.data.DistributedSampler(
        #             dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        #         sampler_test = torch.utils.data.DistributedSampler(
        #             dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        #     else:
        #         sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        #         sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        # else:
        #     sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        #     sampler_test = torch.utils.data.SequentialSampler(dataset_test)
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

        self.model = copy.deepcopy(self.search_space.model)
        self.model.to(self.device)
        # self.model_without_ddp = self.model
        # if config.DISTRIBUTED:
        #     self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[config.GPU])
        #     self.model_without_ddp = self.model.module
        # self.model_without_ddp.load_state_dict(torch.load(config.MODELPATH)['model'])
        self.model.load_state_dict(torch.load(config.MODELPATH)['model'])

    @torch.no_grad()
    def evaluate(self, data_loader, amp=True, retrain_config=None):
        metric_logger = MetricLogger(delimiter="  ")
        header = 'Test:'

        self.model.eval()
        config = self.search_space.decode(retrain_config)
        model_module = unwrap_model(self.model)
        model_module.set_sample_config(config=config)

        # logging.info("sampled model config: {}".format(config))
        # parameters = model_module.get_sampled_params_numel(config)
        # logging.info("sampled model parameters: {}".format(parameters))

        # for images, target in metric_logger.log_every(data_loader, 10, header):
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
        # gather the stats from all processes
        # metric_logger.synchronize_between_processes()
        # logging.info('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
        #     .format(top1=metric_logger.acc1, top5=metric_logger.acc5))

        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


    def __call__(self, arch):
        logging.info("Sampled arch: {}".format(arch))
        val_res = self.evaluate(self.val_loader, amp=self.config.AMP, retrain_config=arch)
        test_res = self.evaluate(self.test_loader, amp=self.config.AMP, retrain_config=arch)
        logging.info("Score: {}\tPerformence: {}".format(val_res, test_res))
        return val_res['acc1'], test_res['acc1']
