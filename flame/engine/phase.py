from torch.utils.data import DataLoader


class Phase(object):
    def __init__(self, name: str, loader: DataLoader):
        self._name = name
        self._loader = loader

    @property
    def name(self) -> str:
        return self._name

    @property
    def loader(self) -> DataLoader:
        return self._loader

    def __eq__(self, other):
        return all([
            self.name == other.name,
            id(self.loader) == id(other.loader)
        ])

    def __hash__(self):
        return id(self)

    def __str__(self):
        return "Phase({})".format(self.name)
