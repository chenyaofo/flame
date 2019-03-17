import typing
from flame.engine import BaseContext
from numbers import Number

T = typing.TypeVar("T")


class ValueHolder(object):
    def __init__(self, initial_value: T, compare: typing.Callable[[T, T], typing.Any] = None,
                 attribute_map: typing.Callable[[BaseContext], typing.Any] = None):
        self.value = initial_value
        self.compare = compare
        self.attribute_map = attribute_map

    def __call__(self, ctx) -> bool:
        self.value, flag = self.compare(self.attribute_map(ctx), self.value)
        return flag


class LargerValueHolder(ValueHolder):
    def __init__(self, initial_value: T = float("-inf"),
                 attribute_map: typing.Callable[[BaseContext], typing.Any] = None):
        super(LargerValueHolder, self).__init__(
            initial_value=initial_value,
            compare=lambda a, b: (a if a > b else b, a > b),
            attribute_map=attribute_map
        )


class SmallerValueHolder(ValueHolder):
    def __init__(self, initial_value: T = float("inf"),
                 attribute_map: typing.Callable[[BaseContext], typing.Any] = None):
        super(SmallerValueHolder, self).__init__(
            initial_value=initial_value,
            compare=lambda a, b: (a if a < b else b, a < b),
            attribute_map=attribute_map
        )
