"""Configuration pytest commune."""
from __future__ import annotations


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: tests longs (chargent CREPE / TensorFlow). Lancer avec -m slow.",
    )
