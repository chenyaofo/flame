import time

from .handler import Handler
from flame.engine import Engine, Event, BaseContext


class TimeEstimater(Handler):
    def __init__(self, name: str):
        super(TimeEstimater, self).__init__(name)
        self.total = None
        self.current = None

    def reset(self, *args, **kwargs) -> None:
        self.start = time.time()

    def update(self, current) -> None:
        self.current = current

    def set_total(self, total) -> None:
        self.total = total

    def _engine_update(self, engine: Engine, ctx: BaseContext) -> None:
        self.update(ctx.iteration)

    def _engine_set_total(self, engine: Engine, ctx: BaseContext) -> None:
        self.set_total(ctx.max_iteration)

    @property
    def value(self) -> float:
        now = time.time()
        remain = (now - self.start) / self.current * (self.total - self.current)
        return remain

    def attach(self, engine: Engine) -> None:
        engine.ctx.entrypoints[self.name] = self
        engine.add_event_handler(Event.PHASE_STARTED, self.reset, self._engine_set_total)
        engine.add_event_handler(Event.ITER_COMPLETED, self._engine_update)
