import time
from collections import OrderedDict


class Timer(object):
    def __init__(self):
        self.start = time.perf_counter()
        self.end = None
        self.record = OrderedDict()

    def mark(self, name=None) -> float:
        self.end = time.perf_counter()
        interval = self.end - self.start
        self.start = self.end
        if name is not None:
            self.record[name] = interval
        return interval


class IterFunctionContainer(object):
    def __init__(self):
        self.regular_version = None
        self.debug_version = None

    def get(self, debug: bool):
        if debug:
            if self.debug_version is not None:
                return self.debug_version
        if self.regular_version is not None:
            return self.regular_version
        raise RuntimeError()

    def set(self, debug: bool, value):
        if debug:
            if self.debug_version is None:
                self.debug_version = value
                return
        else:
            if self.regular_version is None:
                self.regular_version = value
                return
        raise ValueError()
