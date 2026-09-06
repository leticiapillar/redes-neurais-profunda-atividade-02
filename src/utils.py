"""Utilitários: reprodutibilidade, estatísticas do dataset e visualizações."""
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_mean_std(manifest_df, image_size: int = 128, sample_size: int = 400):
    """Calcula média/desvio-padrão por canal em uma amostra do conjunto de treino."""
    sample = manifest_df.sample(min(sample_size, len(manifest_df)), random_state=42)
    pixels = []
    for _, row in sample.iterrows():
        img = Image.open(PROJECT_ROOT / row.image_path).convert("RGB").resize((image_size, image_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        pixels.append(arr.reshape(-1, 3))
    pixels = np.concatenate(pixels, axis=0)
    mean = tuple(pixels.mean(axis=0).tolist())
    std = tuple(pixels.std(axis=0).tolist())
    return mean, std


def plot_curves(history: dict, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history["train_loss"], label="treino")
    axes[0].plot(history["val_loss"], label="validação")
    axes[0].set_title("Curva de perda (BCE + Dice)")
    axes[0].set_xlabel("época")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["val_dice"], label="Dice (val, todas as fatias)", color="green", alpha=0.5)
    if "val_dice_tumor" in history:
        axes[1].plot(history["val_dice_tumor"], label="Dice (val, só c/ tumor)", color="darkgreen")
    axes[1].plot(history["val_iou"], label="IoU (val, todas as fatias)", color="orange", alpha=0.5)
    if "val_iou_tumor" in history:
        axes[1].plot(history["val_iou_tumor"], label="IoU (val, só c/ tumor)", color="darkorange")
    axes[1].set_title("Métricas de validação")
    axes[1].set_xlabel("época")
    axes[1].set_ylabel("score")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def denormalize(image_tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return (image_tensor * std + mean).clamp(0, 1)


def plot_prediction_grid(samples, out_path: Path, mean=None, std=None):
    """samples: lista de tuplas (image_tensor, mask_tensor, pred_tensor)."""
    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[None, :]

    for i, (image, mask, pred) in enumerate(samples):
        if mean is not None:
            image = denormalize(image, mean, std)
        img_np = image.permute(1, 2, 0).numpy()

        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title("Imagem (MRI)" if i == 0 else "")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(img_np)
        axes[i, 1].imshow(mask.squeeze(0).numpy(), cmap="Reds", alpha=0.45)
        axes[i, 1].set_title("Máscara real (overlay)" if i == 0 else "")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(img_np)
        axes[i, 2].imshow(pred.squeeze(0).numpy(), cmap="Reds", alpha=0.45)
        axes[i, 2].set_title("Predição (overlay)" if i == 0 else "")
        axes[i, 2].axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
