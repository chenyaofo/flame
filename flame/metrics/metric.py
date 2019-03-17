import typing

from flame._utils import check
from flame.engine import Engine, Event, BaseContext


class Metric(object):
    def __init__(self, name: str, grad_enabled: bool, context_map: typing.Callable[[BaseContext], typing.Any] = None,
                 reset_event: Event = Event.PHASE_STARTED, update_event: Event = Event.ITER_COMPLETED,
                 multi_update_params: bool = False):
        self.name = check(name, "name", str)
        self.grad_enabled = check(grad_enabled, "grad_enabled", bool)
        self.context_map = check(context_map, "context_map", None,
                                 condition=lambda context_map: context_map is None or callable(context_map))
        self.reset_event = check(reset_event, "reset_event", Event)
        self.update_event = check(update_event, "update_event", Event)

        self.multi_update_params = multi_update_params

    def _check_context_map(self):
        if self.context_map is None:
            raise ValueError("The context_map should be a function if the metric needs to attach an engine.")

    def reset(self, engine: Engine, ctx: BaseContext) -> None:
        pass

    def update(self, *args, **kwargs) -> None:
        pass

    def _engine_update(self, engine: Engine, ctx: BaseContext):
        if self.multi_update_params:
            self.update(*self.context_map(ctx))
        else:
            self.update(self.context_map(ctx))

    def attach(self, engine: Engine) -> None:
        self._check_context_map()
        engine.ctx.entrypoints[self.name] = self
        engine.add_event_handler(self.reset_event, self.reset)
        engine.add_event_handler(self.update_event, self._engine_update)

    @property
    def value(self) -> typing.Any:
        raise NotImplementedError()
