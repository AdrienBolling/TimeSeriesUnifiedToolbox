"""Placeholder for common Mixin stuff."""
import uuid

from pydantic import PrivateAttr
from pydantic_settings import BaseSettings


class MixinSettings(BaseSettings):
    """Base configuration for Mixins."""

    _id: uuid.UUID = PrivateAttr(uuid.uuid4())
