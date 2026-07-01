try:
    import matplotlib
    # Use the non-interactive Agg backend for headless/script runs, but DON'T
    # clobber a notebook's inline/interactive backend (e.g. one selected via
    # %matplotlib inline before importing this package) — overriding it to Agg
    # silently breaks inline figure rendering (plt.show() becomes a no-op).
    if "inline" not in matplotlib.get_backend().lower():
        matplotlib.use("Agg")
except ImportError:
    pass

from .common import Device
from .model import ModelSpec
from .network import Network
from .simulator import Simulator
from .evaluate import run_latency_evaluation

__all__ = [
    "Device",
    "ModelSpec",
    "Network",
    "Simulator",
    "run_latency_evaluation",
]
