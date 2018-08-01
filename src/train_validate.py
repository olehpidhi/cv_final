import torch


def train_epoch(train_loader, model, criterion, optimizer, epoch, device, log, timer):
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    # number of training samples
    num_iter = len(train_loader)

    for i, (input_data, target) in enumerate(train_loader):
        target = target.to(device)
        input_data = input_data.to(device)

        # compute output
        output = model(input_data)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output, target)
        losses.update(loss, input_data.size(0))
        top1.update(prec1[0], input_data.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # logging
        time_string = timer.get_current_str()  # get current time
        log.log_iter(i, epoch, num_iter, losses.val.to('cpu').item(), time_string)

    return top1.avg.to('cpu').item(), losses.avg.to('cpu').item()


def validate(val_loader, model, criterion, device):
    top1 = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input_data, target) in enumerate(val_loader):
            target = target.to(device)
            input_data = input_data.to(device)

            # compute output
            output = model(input_data)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1 = accuracy(output, target)
            losses.update(loss, input_data.size(0))
            top1.update(prec1[0], input_data.size(0))

    return top1.avg.to('cpu').item(), losses.avg.to('cpu').item()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

    return res
