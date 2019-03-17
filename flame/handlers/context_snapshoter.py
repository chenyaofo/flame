import os
import torch
import typing

from flame._utils import check
from flame import output_directory
from flame.handlers.handler import Handler
from flame.engine import Engine, BaseContext, Event


class ContextSnapshot(Handler):
    def __init__(self, name: str, excute_if: typing.Callable[[Engine, BaseContext], bool],
                 store_directory: str = output_directory, prefix: str = "context",
                 exclude: bool = True, excute_event: Event = Event.PHASE_COMPLETED):
        super(ContextSnapshot, self).__init__(name)
        self.excute_if = check(excute_if, "excute_if", None, lambda f: callable(f))
        self.store_directory = check(store_directory, "store_directory", str)
        self.prefix = check(prefix, "prefix", str)
        self.exclude = check(exclude, "exclude", bool)
        self.excute_event = check(excute_event, "excute_event", Event)
        self.last_save = None

    def excute(self, engine: Engine, ctx: BaseContext) -> None:
        if self.excute_if(ctx):
            pure_name = "{}@epoch={}.snapshot".format(self.prefix, ctx.epoch)
            save_path = os.path.join(self.store_directory, pure_name)
            torch.save(ctx.state_dict(), save_path)

            if self.exclude and self.last_save is not None:
                os.remove(self.last_save)
            self.last_save = save_path

    def attach(self, engine: Engine) -> None:
        engine.ctx.plugins[self.name] = self
        engine.add_event_handler(self.excute_event, self.excute)
