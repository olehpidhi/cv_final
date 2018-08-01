from argparse import ArgumentParser
import yaml

import torch.nn.functional as F
from torch.optim import SGD

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import CategoricalAccuracy, Loss

from logger import Logger
from timer import Timer
from models.model_factory import ModelFactory
from datasets.ds_factory import DatasetFactory
from utils import handle_device, next_expr_name, load_checkpoint, save_checkpoint


def train(params, log, time_keeper):
    # specify dataset
    dataset = DatasetFactory.create(params)

    # specify model
    model = ModelFactory.create(params)
    model = model.to(params['device'])

    # optiimizer
    optimizer = SGD(model.parameters(),
                    lr=params['TRAIN']['lr'],
                    momentum=params['TRAIN']['momentum'])

    # scheduler
    scheduler = None

    # best accuracy(precision)
    best_prec = 0

    # optionally resume from a checkpoint
    checkpoint_file = params['TRAIN']['resume']
    start_epoch, best_prec = load_checkpoint(log, model, checkpoint_file, optimizer, scheduler)

    trainer = create_supervised_trainer(model,
                                        optimizer,
                                        F.cross_entropy,
                                        device=params['device'])

    # evaluator
    evaluator = create_supervised_evaluator(model,
                                            metrics={'accuracy': CategoricalAccuracy(),
                                                     'cross_entropy': Loss(F.cross_entropy)},
                                            device=params['device'])
    # log details
    log_string = "\n" + "==== NET MODEL:\n" + str(model)
    log_string += "\n" + "==== OPTIMIZER:\n" + str(optimizer) + "\n"
    log.log_global(log_string)

    # end-of-iteration events
    @trainer.on(Events.ITERATION_COMPLETED)
    def on_iter(engine):
        iter_current = engine.state.iteration % len(dataset.loader['train'])
        epoch_current = engine.state.epoch
        num_iter = len(dataset.loader['train'])
        loss = engine.state.output

        # logging
        time_string = time_keeper.get_current_str()  # get current time
        log.log_iter(iter_current, epoch_current - 1, num_iter, loss, time_string)

    # end-of-epoch events
    @trainer.on(Events.EPOCH_COMPLETED)
    def on_epoch(engine):
        nonlocal best_prec

        # current epoch
        epoch_current = engine.state.epoch

        # evaluation on train set
        evaluator.run(dataset.loader['train'])
        acc_train = evaluator.state.metrics['accuracy'] * 100
        loss_train = evaluator.state.metrics['cross_entropy']

        # evaluation on val set
        evaluator.run(dataset.loader['val'])
        acc_val = evaluator.state.metrics['accuracy'] * 100
        loss_val = evaluator.state.metrics['cross_entropy']

        is_best = acc_val > best_prec
        best_prec = max(acc_val, best_prec)
        save_checkpoint({
            'epoch': epoch_current + 1,
            'state_dict': model.state_dict(),
            'best_prec': best_prec,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler
        }, model, params, is_best)

        # logging results
        time_string = time_keeper.get_current_str()  # get current time
        log.log_epoch(epoch_current,
                      acc_train, loss_train,
                      acc_val, loss_val,
                      is_best, time_string)

    time_keeper.start()
    trainer.run(dataset.loader['train'], max_epochs=params['TRAIN']['epochs'])


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
        params['experiment_name'] = next_expr_name(params['LOG']['path'], "e", 4)

    # manage gpu/cpu devices
    params['device'] = handle_device(params['with_cuda'])

    # logging
    logger = Logger(params)

    # time_keeper
    time_keeper = Timer("global")

    # train
    train(params, logger, time_keeper)

    # close all files
    logger.close()
