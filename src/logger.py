import numpy as np
import os
import datetime


def create_plot_window(vis, xlabel, ylabel, title, env):
    return vis.line(X=np.array([1]), Y=np.array([np.nan]), env=env,
                    opts=dict(xlabel=xlabel, ylabel=ylabel, title=title))


def create_plot_window_two(vis, xlabel, ylabel, title, legends, env):
    return vis.line(X=np.array([[1, 1]]), Y=np.array([[np.nan, np.nan]]), env=env,
                    opts=dict(xlabel=xlabel, ylabel=ylabel, title=title, markers=['o', 'o'], legend=legends))


def dict2str(d, start_n=0):
    res = ""
    prefix_val = " " * start_n
    for k in d:
        if type(d[k]) is dict:
            res += prefix_val + str(k) + ": " + "\n" + dict2str(d[k], start_n + 2)
        else:
            res += prefix_val + str(k) + ": " + str(d[k]) + "\n"
    return res


class Logger(object):
    def __init__(self, params):

        self.with_visdom = params['LOG']['visdom']
        self.experiment_name = params['experiment_name']
        self.iter_interval = params['LOG']['iter_interval']

        path_log_files = os.path.join(params['path_save'], self.experiment_name)

        # log files names
        filename_log_epoch = path_log_files + "_log_epoch.txt"
        filename_log_iter = path_log_files + "_log_iter.txt"
        filename_global = path_log_files + "_log.txt"

        # create and open log files
        self.f_log_iter = open(filename_log_iter, "w+")
        self.f_log_epoch = open(filename_log_epoch, "w+")
        self.f_log_global = open(filename_global, "w+")

        # init all files
        self.f_log_iter.write("{:>6} {:>14} {:>14}\n".format('iter', 'loss', 'elapsed_time'))
        self.f_log_epoch.write("{:>6} {:>14} {:>14} {:>14} {:>14} {:>14}\n".
                               format('epoch',
                                      'avg_acc_train', 'avg_loss_train', 'avg_acc_val', 'avg_loss_val',
                                      'elapsed_time'))

        now = datetime.datetime.now()
        str2log = str(now) + "\n\n" + "==== PARAMETERS:\n" + dict2str(params)
        self.log_global(str2log)

        # make visdom logging
        if self.with_visdom:
            # TODO: check weather this is a good style
            import visdom

            self.vis = visdom.Visdom()
            if not self.vis.check_connection():
                raise RuntimeError("Visdom server not running. Please run python -m visdom.server")

            self.vis.close(win=None, env=self.experiment_name)

            self.train_loss_window = create_plot_window(self.vis, '#Iterations', 'Loss',
                                                        'Training Loss', env=self.experiment_name)
            self.avg_loss_window = create_plot_window_two(self.vis, '#Epochs', 'Loss',
                                                          'Average Loss', ['train', 'test'],
                                                          env=self.experiment_name)
            self.avg_acc_window = create_plot_window_two(self.vis, '#Epochs', 'Accuracy',
                                                         'Average Accuracy', ['train', 'test'],
                                                         env=self.experiment_name)

    def log_iter(self, iter_current, epoch_current, num_iter, loss, time_str):

        if iter_current % self.iter_interval == 0:

            # log details
            log_string = "Epoch[{}] Iteration[{}/{}] Loss: {:.2f} Time: {}" \
                         "".format(epoch_current, iter_current, num_iter, loss, time_str)
            self.log_global(log_string)

            self.f_log_iter.write(
                "{:6d} {:14.4f} {:>14}\n".format(iter_current + epoch_current * num_iter, loss, time_str))

            # visdom log
            if self.with_visdom:
                self.vis.line(X=np.array([iter_current + epoch_current * num_iter]),
                              Y=np.array([loss]),
                              update='append', win=self.train_loss_window, env=self.experiment_name)

    def log_epoch(self,
                  n_epoch,
                  acc_train, loss_train, acc_val, loss_val, is_best, time_str):

        # log details
        log_string = "Epoch: {}  Train Avg accuracy: {:.2f} Train  Avg loss: {:.2f}" \
                     " Validation Avg accuracy: {:.2f} Validation Avg loss: {:.2f} Time: {}" \
                     "".format(n_epoch, acc_train, loss_train,
                               acc_val, loss_val, time_str) + \
                     (" BEST MODEL SAVED" if is_best else "")
        self.log_global(log_string)

        self.f_log_epoch.write("{:6d} {:14.4f} {:14.4f} {:14.4f} {:14.4f} {:>14}\n".
                               format(n_epoch, acc_train, loss_train, acc_val, loss_val, time_str))

        # make visdom logging
        if self.with_visdom:
            self.vis.line(X=np.array([[n_epoch, n_epoch]]),
                          Y=np.array([[loss_train, loss_val]]),
                          opts=dict(legend=['train', 'test']),
                          win=self.avg_loss_window, update='append', env=self.experiment_name)
            self.vis.line(X=np.array([[n_epoch, n_epoch]]),
                          Y=np.array([[acc_train, acc_val]]),
                          opts=dict(legend=['train', 'test']),
                          win=self.avg_acc_window, update='append', env=self.experiment_name)

    def log_global(self, log_str):
        self.f_log_global.write(log_str + "\n")
        print(log_str)

    def close(self):
        # close log files
        self.f_log_iter.close()
        self.f_log_epoch.close()
        self.f_log_global.close()
