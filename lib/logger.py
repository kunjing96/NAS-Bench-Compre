import os
import sys
import copy
import logging
import PIL
import torch


def prepare_logger(config):
    config = copy.deepcopy(config)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(config.OUTPUT, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info('Main Function with logger : {:}'.format(logging))
    logging.info('Arguments : -------------------------------')
    logging.info("{:}".format(config))
    logging.info("Python  Version  : {:}".format(sys.version.replace('\n', ' ')))
    logging.info("Pillow  Version  : {:}".format(PIL.__version__))
    logging.info("PyTorch Version  : {:}".format(torch.__version__))
    logging.info("cuDNN   Version  : {:}".format(torch.backends.cudnn.version()))
    logging.info("CUDA available   : {:}".format(torch.cuda.is_available()))
    logging.info("CUDA GPU numbers : {:}".format(torch.cuda.device_count()))
    logging.info("CUDA_VISIBLE_DEVICES : {:}".format(os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ else 'None'))