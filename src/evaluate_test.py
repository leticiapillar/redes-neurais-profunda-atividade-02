"""Avalia o melhor checkpoint no conjunto de TESTE (holdout, nunca visto no treino/val)."""
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import BrainMRISegmentationDataset
from engine import evaluate
from losses import BCEDiceLoss, dice_coefficient, iou_score
from model import UNet
from utils import plot_prediction_grid, set_seed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT_ROOT / "outputs"


def main():
    set_seed(42)
    device = torch.device("cpu")

    ckpt = torch.load(OUTPUTS / "checkpoints" / "best_model.pt", map_location=device)
    config, mean, std = ckpt["config"], ckpt["mean"], ckpt["std"]

    import pandas as pd

    test_df = pd.read_csv(OUTPUTS / "split_test.csv")

    test_ds = BrainMRISegmentationDataset(test_df, config["image_size"], augment=False, mean=mean, std=std)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)

    model = UNet(base_ch=config["base_ch"], dropout=0.0).to(device)
    model.load_state_dict(ckpt["model_state"])

    loss_fn = BCEDiceLoss(bce_weight=config["bce_weight"], pos_weight=config.get("pos_weight"))
    overall = evaluate(model, test_loader, loss_fn, device, desc="teste (geral)")
    print("Métricas no TESTE (todas as fatias):", overall)

    # Métricas apenas nas fatias que de fato contêm tumor (mais informativo,
    # já que fatias sem tumor tendem a inflar o Dice/IoU médio com acertos triviais).
    tumor_df = test_df[test_df.has_tumor].reset_index(drop=True)
    tumor_ds = BrainMRISegmentationDataset(tumor_df, config["image_size"], augment=False, mean=mean, std=std)
    tumor_loader = DataLoader(tumor_ds, batch_size=config["batch_size"], shuffle=False)
    tumor_only = evaluate(model, tumor_loader, loss_fn, device, desc="teste (só c/ tumor)")
    print("Métricas no TESTE (só fatias com tumor):", tumor_only)

    metrics = {"overall": overall, "tumor_slices_only": tumor_only, "n_test_slices": len(test_df), "n_tumor_slices": len(tumor_df)}
    with open(OUTPUTS / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Grade de visualização: 3 fatias com tumor + 2 sem tumor.
    model.eval()
    samples = []
    rng = torch.Generator().manual_seed(0)

    chosen_tumor = tumor_df.sample(min(3, len(tumor_df)), random_state=0)
    chosen_empty = test_df[~test_df.has_tumor].sample(min(2, len(test_df[~test_df.has_tumor])), random_state=0)
    chosen = pd.concat([chosen_tumor, chosen_empty])

    vis_ds = BrainMRISegmentationDataset(chosen, config["image_size"], augment=False, mean=mean, std=std)
    with torch.no_grad():
        for i in range(len(vis_ds)):
            image, mask = vis_ds[i]
            logits = model(image.unsqueeze(0))
            pred = (torch.sigmoid(logits) > 0.5).float().squeeze(0)
            samples.append((image, mask, pred))

    plot_prediction_grid(samples, OUTPUTS / "figures" / "test_predictions.png", mean=mean, std=std)
    print("Figura salva em outputs/figures/test_predictions.png")


if __name__ == "__main__":
    main()
