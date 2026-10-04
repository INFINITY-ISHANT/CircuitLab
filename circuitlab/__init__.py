"""Person B's scientific backend for CircuitLab.

Only model loading is working at this stage. Dataset generation, interventions,
evaluation, and accounting are deliberate stubs until scientifically validated.
"""

from .model import get_model

__all__ = ["get_model"]
