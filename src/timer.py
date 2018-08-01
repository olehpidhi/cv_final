import time


class Timer(object):
    def __init__(self, name=None):
        self.name = name
        self.tstart = None

    def start(self):
        self.tstart = time.time()

    def get_current(self):
        return time.time() - self.tstart

    def get_current_str(self):
        time_diff = time.time() - self.tstart
        days, rem = divmod(time_diff, 24 * 60 * 60)
        hours, rem = divmod(rem, 60 * 60)
        minutes, seconds = divmod(rem, 60)
        return "{:0>2}:{:0>2}:{:0>2}:{:0>2}".format(int(days), int(hours), int(minutes), int(seconds))
