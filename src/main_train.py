from argparse import ArgumentParser
import yaml

import torch
import torch.nn as nn

from logger import Logger
from timer import Timer

from models.model_factory import ModelFactory
from datasets.ds_factory import DatasetFactory

from utils import handle_device, next_expr_name, load_checkpoint, save_checkpoint
from train_validate import train_epoch, validate


def train(params, log, time_keeper):
    # specify dataset
    data = DatasetFactory.create(params)

    # specify model
    model = ModelFactory.create(params)
    model = model.to(params['device'])

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(params['device'])
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=params['TRAIN']['lr'],
                                momentum=params['TRAIN']['momentum'])

    # resume from a checkpoint
    if len(params['TRAIN']['resume']) > 0:
        start_epoch, best_prec = load_checkpoint(log, model, params['TRAIN']['resume'], optimizer)
    else:
        start_epoch = 0
        best_prec = 0

    # scheduler (if any)
    if 'lr_schedule_step' in params['TRAIN']:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=params['TRAIN']['lr_schedule_step'],
                                                    gamma=params['TRAIN']['lr_schedule_gamma'])
    else:
        scheduler = None

    # log details
    log_string = "\n" + "==== NET MODEL:\n" + str(model)
    log_string += "\n" + "==== OPTIMIZER:\n" + str(optimizer) + "\n"
    log.log_global(log_string)

    time_keeper.start()

    # train
    for epoch in range(start_epoch, params['TRAIN']['epochs']):
        # adjust_learning_rate
        if scheduler:
            scheduler.step()

        # train for one epoch
        _, _ = train_epoch(data.loader['train'], model, criterion, optimizer, epoch,
                           params['device'], log, timer)

        # evaluate on train set
        acc_train, loss_train = validate(data.loader['train'], model, criterion, params['device'])

        # evaluate on validation set
        acc_val, loss_val = validate(data.loader['val'], model, criterion, params['device'])

        # remember best prec@1 and save checkpoint
        is_best = acc_val > best_prec
        best_prec = max(acc_val, best_prec)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec': best_prec,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler
        }, model, params, is_best)

        # logging results
        time_string = time_keeper.get_current_str()  # get current time
        log.log_epoch(epoch + 1,
                      acc_train, loss_train,
                      acc_val, loss_val,
                      is_best, time_string)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--param_file', type=str,
                        help='configure file with parameters')

    args = parser.parse_args()

    # parse param file
    with open(args.param_file, 'r') as f:
        params = yaml.load(f)

    # experiment name
    if len(params['experiment_name']) == 0:
        params['experiment_name'] = next_expr_name(params['path_save'], "e", 4)

    # manage gpu/cpu devices
    params['device'] = handle_device(params['with_cuda'])

    # logging
    logger = Logger(params)

    # timer
    timer = Timer("global")

    # train
    train(params, logger, timer)

    # close all log files
    logger.close()
