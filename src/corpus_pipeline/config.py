"""
config.py
---------
Loads and validates config/config.yaml.
Provides a single `cfg` object used by every other module.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "config.yaml"


class Config:
    """Thin wrapper around the YAML dict with attribute-style access."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    def __getattr__(self, key: str) -> Any:
        try:
            val = self._data[key]
        except KeyError:
            raise AttributeError(f"Config has no key '{key}'") from None
        if isinstance(val, dict):
            return Config(val)
        return val

    def get(self, key: str, default: Any = None) -> Any:
        val = self._data.get(key, default)
        if isinstance(val, dict):
            return Config(val)
        return val

    def __repr__(self) -> str:  # pragma: no cover
        return f"Config({list(self._data.keys())})"


def load(path: str | Path | None = None) -> Config:
    """Load and return the pipeline config."""
    from pathlib import Path  # Move import to top of function
    
    if path is None:
        config_path = Path(__file__).parent.parent.parent / 'config' / 'config.yaml'
    else:
        config_path = Path(path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return Config(data)


# Compile-once helper used by other modules
def compile_pattern(pattern_str: str, flags: int = re.MULTILINE) -> re.Pattern:
    return re.compile(pattern_str, flags)