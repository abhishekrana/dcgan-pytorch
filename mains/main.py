import os
import pudb
import json
import logging
import argparse
import numpy as np
import time
import copy
import pickle
import glob
import datetime
import random
import pudb
import torch
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

os.sys.path.append('./')
os.sys.path.append('../')
import utils.util as utils
from data_handlers.data_handler import DataHandler
from models.model import Model
from trainers.trainer import Trainer



def main(config, resume):

    ## Set config paths
    config['timestamp_start'] = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
    config['output_path'] = os.path.join(config['output_dir'], config['name'], config['timestamp_start'] + '_result')
    config['output_images_path'] = os.path.join(config['output_dir'], config['name'], config['timestamp_start'] + '_result', 'images')
    config['checkpoint_path'] = os.path.join(config['output_path'], 'checkpoints')
    config['summary_path'] = os.path.join(config['output_path'], 'summary')


    ## Create output dirs
    # utils.remove_dirs([os.path.join(config.output_path, config.exp_name)])
    utils.create_dirs([config['output_path'], config['output_images_path'], config['checkpoint_path'], config['summary_path']])


    ## Logger
    # utils.logger_init(config, logging.DEBUG)
    utils.logger_init(config, logging.INFO)
    logging.info('config {}'.format(config))


    ## Register signal handler
    # utils.signal_handler(config)


    ## Set config variables
    use_cuda = torch.cuda.is_available() and config['cuda']
    config['device'] = torch.device('cuda:0' if use_cuda else 'cpu')
    config['writer'] = SummaryWriter(config['summary_path'])


    ## Set seed values to reproduce results
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if config['device'] != 'cpu':
        torch.cuda.manual_seed(config['seed'])
    # cuDNN uses nondeterministic algorithms, and it can be disabled using 
    # torch.backends.cudnn.enabled = False

    # It enables benchmark mode in cudnn.
    # Use only when input sizes for your network do not vary. 
    # This way, cudnn will look for the optimal set of algorithms for that particular 
    # configuration (which takes some time). This usually leads to faster runtime.
    # But if input sizes changes at each iteration, then cudnn will benchmark 
    # every time a new size appears, possibly leading to worse runtime performances.
    cudnn.benchmark = True

    ## Save code
    # utils.save_code(config)

    # size of the latent z vector
    # config['nz'] = 100

    ## Data
    data_handler = DataHandler(config)


    ## Model
    model = Model(config)


    ## Train
    trainer = Trainer(config, data_handler.dataloaders, model)
    trainer.train()

    ## Test
    # trainer.test()

    config['timestamp_end'] = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
    logging.info('timestamp_end {}'.format(config['timestamp_end']))


if __name__ == '__main__':
    args = utils.get_args()

    config = None
    if args.resume is not None:
        if args.config is not None:
            logger.warning('Warning: --config overridden by --resume')
        config = torch.load(args.resume)['config']
    elif args.config is not None:
        config = json.load(open(args.config))
        path = os.path.join(config['trainer']['save_dir'], config['name'])
        # assert not os.path.exists(path), "Path {} already exists!".format(path)
    assert config is not None

    main(config, args.resume)
