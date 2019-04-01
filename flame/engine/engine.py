import typing
from collections import defaultdict

import flame
from flame.engine.context import BaseContext
from flame.engine.event import Event
from flame.engine.phase import Phase
from flame.engine.exception import EngineStopException, EpochStopException, PhaseStopException, IterationStopException
from .utils import Timer, IterFunctionContainer


class Engine(object):
    def __init__(self, ctx: BaseContext, debug=flame.debug, logger=flame.logger):
        self.ctx = ctx
        self.debug = debug
        self.logger = logger
        if self.debug:
            self.logger.debug("The engine({}) is running with debug mode.".format(hex(id(self))))

        self._iter_funcs = defaultdict(IterFunctionContainer)

        self._event_handlers = defaultdict(list)
        self._debug_event_handlers = defaultdict(list)

        self._epoch_flow_control = None

    def iter_func(self, phase: Phase, debug: bool = False):

        def decorator(f):
            self._iter_funcs[phase].set(debug, f)
            return f

        return decorator

    def on(self, event: Event, debug: bool = False):

        def decorator(f):
            if debug:
                self.add_debug_event_handler(event, f)
            else:
                self.add_event_handler(event, f)
            return f

        return decorator

    @property
    def epoch_flow_control(self):
        def decorator(f):
            self._epoch_flow_control = f
            return f

        return decorator

    def stop_engine(self) -> typing.NoReturn:
        raise EngineStopException()

    def skip_epoch(self) -> typing.NoReturn:
        raise EpochStopException()

    def skip_phase(self) -> typing.NoReturn:
        raise PhaseStopException()

    def skip_iter(self) -> typing.NoReturn:
        raise IterationStopException()

    def run(self) -> None:
        try:
            self._trigger_event(Event.ENGINE_STARTED)
            self.run_epoch()
        except EngineStopException:
            self.logger.debug("The engine({}) is early stopped.".format(hex(id(self))))
        finally:
            self._trigger_event(Event.ENGINE_COMPLETED)

    def run_epoch(self) -> None:
        try:
            while self.ctx.epoch < self.ctx.max_epoch:
                self.ctx.epoch += 1
                self._trigger_event(Event.EPOCH_STARTED)
                self._epoch_flow_control(self, self.ctx)
        except EpochStopException:
            self.logger.debug("The epoch({}) is early stopped.".format(self.ctx.epoch))
        finally:
            self._trigger_event(Event.EPOCH_COMPLETED)
            self.ctx.clean_epoch_variable()

    def run_phase(self, phase: Phase) -> None:
        if not self.ctx.is_register_phase(phase):
            raise ValueError("The engine({}) fails to run the phase({}) because it is not registered."
                             .format(hex(id(self)), phase.name))
        self.ctx.phase = phase

        loader = phase.loader
        self.ctx.iteration = 0
        self.ctx.max_iteration = len(loader)

        self._trigger_event(Event.PHASE_STARTED)
        try:
            if self.debug:
                loader_iter = iter(loader)
                while True:
                    self.ctx.timer = Timer()
                    self.ctx.inputs = next(loader_iter)
                    self.ctx.timer.mark("data loading")
                    self.run_iter(phase)

                    self.logger.debug("Time perf statistic (iter={}): {}."
                                      .format(self.ctx.iteration,
                                              ", ".join(["{}: {:.4f}s"
                                                        .format(name, t) for name, t in
                                                         self.ctx.timer.record.items()])))
            else:
                for self.ctx.inputs in loader:
                    self.run_iter(phase)
        except StopIteration:
            pass
        except PhaseStopException:
            self.logger.debug("The phase({}) is early stopped.".format(phase.name))
        finally:
            self._trigger_event(Event.PHASE_COMPLETED)
            self.ctx.clean_phase_variable()

    def run_iter(self, phase: Phase) -> None:
        try:
            self.ctx.iteration += 1
            self._trigger_event(Event.ITER_STARTED)
            iter_func = self._iter_funcs[phase].get(self.debug)
            return iter_func(self, self.ctx)
        except IterationStopException:
            self.logger.debug("The iteration({}) is early stopped.".format(self.ctx.iteration))
        finally:
            self._trigger_event(Event.ITER_COMPLETED)
            self.ctx.clean_iteration_variable()

    def add_event_handler(self, event: Event, *handlers) -> None:
        self._event_handlers[event] += handlers

    def add_debug_event_handler(self, event: Event, *handlers) -> None:
        self._debug_event_handlers[event] += handlers

    def _trigger_event(self, event: Event) -> None:
        self._trigger_debug_event(event)
        self._trigger_regular_event(event)

    def _trigger_regular_event(self, event: Event):
        handlers = self._event_handlers.get(event)
        if handlers is not None:
            for handler in handlers:
                handler(self, self.ctx)

    def _trigger_debug_event(self, event: Event):
        handlers = self._debug_event_handlers.get(event)
        if handlers is not None:
            for handler in handlers:
                handler(self, self.ctx)
