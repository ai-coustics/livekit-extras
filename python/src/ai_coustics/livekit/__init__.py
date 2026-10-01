from importlib.metadata import PackageNotFoundError, version

from aic_sdk import Model
from livekit.agents import Plugin

from .analyzer import AnalysisEvent, Analyzer, Collector
from .log import logger
from .vad import VAD, VADParameters, VADProcessor

try:
    __version__ = version("ai-coustics-livekit-extras")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0.0.0"


class AICousticsPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__ or __name__, logger)


Plugin.register_plugin(AICousticsPlugin())

__all__ = [
    "AICousticsPlugin",
    "AnalysisEvent",
    "Analyzer",
    "Collector",
    "Model",
    "VAD",
    "VADParameters",
    "VADProcessor",
    "__version__",
]
