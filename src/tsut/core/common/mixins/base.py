"""Base class for the general model of a Mixin."""

from abc import ABC
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


class MixinSettings(BaseSettings):
    """Base settings mixin for configuration management."""

    model_config = SettingsConfigDict(env_prefix="TSUT_MIXIN_")


class Mixin(ABC):
    """Base class for Mixins."""

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize the Mixin with settings."""
        if not self._config and "config" in kwargs:
            self._config = kwargs["config"]

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        """Post-initialization hook for additional setup."""
        super().__init__(*args, **kwargs)
        if getattr(super(), "__post_init__", None):
            super().__post_init__(*args, **kwargs)
