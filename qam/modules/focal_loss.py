import torch

from ..constants import PAD_ID
from ..utils import Classifier


class FocalLoss(torch.nn.Module):
    def __init__(
        self,
        alpha: torch.Tensor,
        gamma: float,
        grad_acc: int,
        ignore_index: int = PAD_ID,
        softmax_dim: int = 2,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.grad_acc = grad_acc
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

        loss = torch.true_divide(loss, (count * self.grad_acc))

        return loss
