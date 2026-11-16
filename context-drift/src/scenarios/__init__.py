"""Scenarios: one module per scenario, each exposing ``SCENARIO`` (a ``talkdemo.scenario.Scenario``).

``<name>.json`` next to each module is the same content as data (``python run.py export``):
schema, conventions, examples and questions with their corrupted variants and decision points.
"""

from __future__ import annotations

from importlib import import_module

from talkdemo.scenario import Scenario

NAMES = ("shop", "shop_v1")  # shop_v1: the first version, dropped (see its docstring)


def load(name: str = "shop") -> Scenario:
    return import_module(f"scenarios.{name}").SCENARIO
