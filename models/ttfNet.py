from enum import Enum
from typing import TypedDict, Optional, Type, Callable, Literal, Union
import torch

from torch.nn.functional import logsigmoid, softplus

# nflows dependencies


# custom modules
from models.extreme_transforms import (
    configure_nn,
    NNKwargs,
    TailAffineMarginalTransform,
    flip,
    AffineMarginalTransform,
)











class ModelKwargs(TypedDict, total=False):
    tail_bound: Optional[float]
    num_bins: Optional[int]
    tail_init: Optional[Union[list[float], float]]
    rotation: Optional[bool]
    fix_tails: Optional[bool]
    data: Optional[torch.Tensor]


#######################

class Softplus:
    """
    Softplus non-linearity
    """

    THRESHOLD = 20.0
    EPS = 1e-8  # a small value, to ensure that the inverse doesn't evaluate to 0.

    def __init__(self, offset=1e-3):
        super().__init__()
        self.offset = offset

    def inverse(self, z, context=None):
        # maps real z to postive real x, with log grad
        x = torch.zeros_like(z)
        above = z > self.THRESHOLD
        x[above] = z[above]
        x[~above] = torch.log1p(z[~above].exp())
        lad = logsigmoid(z)
        return self.EPS + x, lad.sum(dim=1)

    def forward(self, x, context=None):
        # if x = 0, little can be done
        if torch.min(x) <= 0:
            raise Exception("Inputs <0 passed to Softplus transform")

        z = x + torch.log(-torch.expm1(-x))
        lad = x - torch.log(torch.expm1(x))

        return z, lad.sum(axis=1)











#######################
# flow models






def build_ttf_m(
    dim: int,
    model_kwargs: ModelKwargs = {},
):
    # configure model specific settings
    pos_tail_init = model_kwargs.get("pos_tail_init", None)
    neg_tail_init = model_kwargs.get("neg_tail_init", None)
    fix_tails = model_kwargs.get("fix_tails", False)


    # set up tail transform
    tail_transform = TailAffineMarginalTransform(
        features=dim,
        pos_tail_init=torch.tensor(pos_tail_init),
        neg_tail_init=torch.tensor(neg_tail_init),
    )

    if fix_tails:
        assert (
            pos_tail_init is not None
        ), "Fixing tails, but no init provided for pos tails"
        assert (
            neg_tail_init is not None
        ), "Fixing tails, but no init provided for neg tails"
        tail_transform.fix_tails()

    # the tail transformation needs to be flipped this means data->noise is
    # a strictly lightening transformation
    tail_transform = flip(tail_transform)

    return tail_transform

