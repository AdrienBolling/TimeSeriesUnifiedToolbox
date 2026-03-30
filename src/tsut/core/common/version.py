from pydantic import BaseModel

class Version(BaseModel):
    """Version information for the TSUT library."""

    major: int
    minor: int
    patch: int

    def __str__(self) -> str:
        """Return the version as a string in the format 'major.minor.patch'."""
        return f"{self.major}.{self.minor}.{self.patch}"