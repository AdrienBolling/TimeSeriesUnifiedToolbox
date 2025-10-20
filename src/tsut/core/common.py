from typing import override
import numpy as np
import pandas as pd
import torch
import ray.data as d
from typing import Union

ArrayLike = Union[
    list, tuple, np.ndarray, pd.Series, pd.DataFrame, torch.Tensor, d.Dataset
]


class EnforcedDict(dict):
    """
    A dict subclass that must contain REQUIRED_KEYS
    - Works where a plain dict is expected (isinstance checks  will pass)
    - Blocks removing these keys and validates after any mutations
    """

    REQUIRED_KEYS: frozenset[str] = frozenset()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        rk = getattr(cls, "REQUIRED_KEYS", frozenset())
        if not isinstance(rk, (set, frozenset)) or not all(
            isinstance(k, str) for k in rk
        ):
            raise TypeError(
                f"{cls.__name__}.REQUIRED_KEYS must be a set/frozenset[str]"
            )
        if not rk:
            raise TypeError(f"{cls.__name__} must define a non-empty REQUIRED_KEYS")

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._validate_required_keys()

    def _validate_required_keys(self) -> None:
        missing_keys = self.REQUIRED_KEYS - self.keys()
        if missing_keys:
            raise KeyError(f"Missing required keys: {missing_keys}")

    # Write operations
    def __setitem__(self, key, value) -> None:
        super().__setitem__(key, value)
        self._validate_required_keys()

    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)
        self._validate_required_keys()

    def setdefault(self, key, default=None) -> None:
        out = super().setdefault(key, default)
        self._validate_required_keys()
        return out

    def pop(self, key, *args):
        if key in self.REQUIRED_KEYS:
            raise KeyError(f"Cannot remove required key: {key}")
        return super().pop(key, *args)

    def popitem(self):
        key, value = super().popitem()
        if key in self.REQUIRED_KEYS:
            raise KeyError(f"Cannot remove required key: {key}")
        return key, value

    def clear(self) -> None:
        if self.REQUIRED_KEYS:
            raise KeyError(
                f"Cannot clear dict with required keys: {self.REQUIRED_KEYS}"
            )
        return super().clear()

    # Py 3.9+ union operator support
    @override
    def __or__(self, other):
        new_dict = super().__or__(other)
        return self.__class__(new_dict)

    @override
    def __ror__(self, other):
        new_dict = super().__ror__(other)
        return self.__class__(new_dict)

    @override
    def __ior__(self, other):
        new_dict = super().__ior__(other)
        return self.__class__(new_dict)

    @override
    def copy(self):
        return self.__class__(self)

    @override
    def to_dict(self) -> dict:
        return dict(self)
