"""lac: what an aggregate accuracy delta hides.

Accuracy is a marginal. Two label sets over the same items can agree on it exactly while
disagreeing about a tenth of the items, and a change that raises it by a point can be made
entirely of items that got worse cancelling items that got better. This package computes the
per-item accounting that the marginal sums away.
"""

from lac.labels import Item, LabelSet
from lac.transitions import Transition, TransitionMatrix

__all__ = ["Item", "LabelSet", "Transition", "TransitionMatrix"]
__version__ = "0.1.0"
