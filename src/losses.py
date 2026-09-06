"""Funções de perda e métricas para segmentação binária."""
import torch
import torch.nn as nn


def dice_coefficient(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = (probs > 0.5).float()
    intersection = (probs * target).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean()


def iou_score(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = (probs > 0.5).float()
    intersection = (probs * target).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean()


def pixel_accuracy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    correct = (preds == target).float().mean()
    return correct


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, target):
        probs = torch.sigmoid(logits)
        intersection = (probs * target).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (2 * intersection + self.eps) / (union + self.eps)
        return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Combinação de Binary Cross-Entropy + Dice Loss.

    BCE fornece gradiente estável pixel a pixel; Dice lida melhor com o forte
    desbalanceamento entre fundo e região de tumor (fatias com tumor pequeno).
    `pos_weight` dá peso extra aos pixels de tumor dentro do termo BCE, já que
    eles representam ~1% dos pixels do dataset.
    """

    def __init__(self, bce_weight: float = 0.5, pos_weight: float | None = None):
        super().__init__()
        pw = torch.tensor(pos_weight) if pos_weight is not None else None
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.dice = DiceLoss()
        self.bce_weight = bce_weight

    def forward(self, logits, target):
        return self.bce_weight * self.bce(logits, target) + (1 - self.bce_weight) * self.dice(logits, target)
