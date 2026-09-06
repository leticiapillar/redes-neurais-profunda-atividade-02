"""Loops de treino e avaliação."""
import torch
from tqdm import tqdm

from losses import dice_coefficient, iou_score, pixel_accuracy


def train_one_epoch(model, loader, optimizer, loss_fn, device, desc="treino"):
    model.train()
    running_loss = 0.0
    for images, masks in tqdm(loader, desc=desc, leave=False):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, desc="validação"):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    running_acc = 0.0
    n_batches = 0
    for images, masks in tqdm(loader, desc=desc, leave=False):
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        loss = loss_fn(logits, masks)
        running_loss += loss.item() * images.size(0)
        running_dice += dice_coefficient(logits, masks).item()
        running_iou += iou_score(logits, masks).item()
        running_acc += pixel_accuracy(logits, masks).item()
        n_batches += 1
    return {
        "loss": running_loss / len(loader.dataset),
        "dice": running_dice / n_batches,
        "iou": running_iou / n_batches,
        "pixel_acc": running_acc / n_batches,
    }
