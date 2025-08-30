from .plugin import Plugin
from .console_out import ConsoleOutPlugin
from .tensorboard import TensorBoardPlugin


__all__ = ["Plugin", "default_plugins", "ConsoleOutPlugin", "TensorBoardPlugin"]


def default_plugins(spec):
    return [ConsoleOutPlugin.init(spec)]
