"""
Weights & Biases plugin for bz CLI.

This plugin provides integration with Weights & Biases for experiment tracking.
"""

from .wandb_plugin import WandBPlugin

__all__ = ["WandBPlugin"]
