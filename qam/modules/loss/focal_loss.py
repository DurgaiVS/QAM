from typing import List, Union

import torch
from omegaconf import DictConfig

from ...constants import PAD_ID
from ...utils import TradeTrend


class FocalLoss(torch.nn.modules.loss._Loss):
    """
    Focal loss class. As described in the paper: https://arxiv.org/abs/1708.02002.
    As per the paper, the loss is defined as:
    L = -∑((1 - p_t)^γ * log(p_t))

    where,
    p_t = labelwise softmaxed logits.
    γ is the focusing parameter (gamma).

    We've modified the loss to include the weight factor for each class, as:
    L = -∑(αi * (1 - p_t)^γ * log(p_t))

    where,
    αi is the weight factor for the class i (alpha).

    Parameters
    ----------
    alpha : torch.Tensor
        Weighting factor for each class.
    gamma : float
        Focusing parameter.
    ignore_index : int, optional
        Index to ignore, by default PAD_LABEL_ID.
    softmax_dim : int, optional
        Dimension for softmax, by default 2.
    """

    def __init__(
        self,
        gamma: float,
        alpha: List[Union[int, float]],
        ignore_index: int = PAD_ID,
        softmax_dim: int = -1,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.softmax_dim = softmax_dim

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ):
        logits = logits.softmax(dim=self.softmax_dim)
        mask = labels != self.ignore_index
        active_labels, active_logits = labels[mask], logits[mask]

        loss = 0.0
        count = torch.numel(active_labels)
        for label in active_labels.unique():
            label_specific_logits = active_logits[active_labels == label][:, label]
            tmp = (
                -(self.alpha[label])
                * ((1 - label_specific_logits) ** self.gamma)
                * (label_specific_logits.log())
            )
            loss += tmp.sum()

        return loss / count

    @classmethod
    def from_cfg(cls, cfg: DictConfig):
        assert ("alpha" in cfg) or (
            ("num_classes" in cfg) and (cfg.num_classes > 0)
        ), "Either `alpha` or `num_classes` should be provided, to initialize the FocalLoss."

        if ("alpha" not in cfg) or (cfg.alpha is None):
            alpha = [1.0 for _ in range(cfg.num_classes)]
        else:
            alpha = cfg.alpha

        return cls(
            alpha=alpha,
            gamma=cfg.gamma,
            softmax_dim=cfg.softmax_dim,
            ignore_index=cfg.get("ignore_index", PAD_ID),
        )
