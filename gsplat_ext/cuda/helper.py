from typing import Optional, Tuple
from torch import Tensor
from ._backend import _C
import torch

def covar_to_quat_scale(
    covars: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Convert covariance matrices to quaternions and scales.

    Args:
        covars: Tensor of covariance matrices: [N, 6]

    Returns:
        Tuple[Tensor, Tensor]: Quaternions and scales: [N, 4], [N, 3]
    """
    if covars.dim() == 3: # input is [N, 3, 3]
        covars = torch.stack([covars[..., 0, 0], covars[..., 0, 1], covars[..., 0, 2], covars[..., 1, 1], covars[..., 1, 2], covars[..., 2, 2]], dim=-1)
    assert covars.dim() == 2 and covars.size(1) == 6, "Covariance matrices must be of shape [N, 6]"
    quats, scales = _C.covar_to_quat_scale(covars)
    return quats, scales
