"""Script principal de treino da U-Net para segmentação de tumor em MRI cerebral."""
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import BrainMRISegmentationDataset
from engine import evaluate, train_one_epoch
from losses import BCEDiceLoss
from manifest import build_manifest
from model import UNet
from splits import patient_level_split
from utils import compute_mean_std, plot_curves, set_seed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT_ROOT / "outputs"

CONFIG = {
    "image_size": 128,
    "batch_size": 32,
    "epochs": 40,
    "lr": 5e-4,
    "weight_decay": 1e-4,
    "base_ch": 16,
    "dropout": 0.1,
    "bce_weight": 0.3,
    "pos_weight": 3.0,
    "early_stop_patience": 10,
    "num_workers": 2,
    "seed": 42,
}


def main():
    set_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device)

    manifest = build_manifest()
    splits = patient_level_split(manifest)
    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]

    print("Calculando média/desvio-padrão do conjunto de treino...")
    mean, std = compute_mean_std(train_df, image_size=CONFIG["image_size"])
    print("mean:", mean, "std:", std)

    val_tumor_df = val_df[val_df.has_tumor].reset_index(drop=True)

    train_ds = BrainMRISegmentationDataset(train_df, CONFIG["image_size"], augment=True, mean=mean, std=std)
    val_ds = BrainMRISegmentationDataset(val_df, CONFIG["image_size"], augment=False, mean=mean, std=std)
    val_tumor_ds = BrainMRISegmentationDataset(val_tumor_df, CONFIG["image_size"], augment=False, mean=mean, std=std)
    test_ds = BrainMRISegmentationDataset(test_df, CONFIG["image_size"], augment=False, mean=mean, std=std)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
    val_tumor_loader = DataLoader(val_tumor_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    model = UNet(base_ch=CONFIG["base_ch"], dropout=CONFIG["dropout"]).to(device)
    loss_fn = BCEDiceLoss(bce_weight=CONFIG["bce_weight"], pos_weight=CONFIG["pos_weight"])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)

    # Seleção do melhor checkpoint e early stopping usam o Dice calculado SOMENTE
    # nas fatias de validação que de fato contêm tumor: o Dice médio em todas as
    # fatias é inflado por fatias vazias (fundo puro), e não reflete a real
    # capacidade do modelo de segmentar a lesão.
    history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_iou": [], "val_dice_tumor": [], "val_iou_tumor": []}
    best_dice_tumor = -1.0
    epochs_no_improve = 0

    OUTPUTS.mkdir(exist_ok=True)
    (OUTPUTS / "checkpoints").mkdir(exist_ok=True)
    (OUTPUTS / "figures").mkdir(exist_ok=True)

    t0 = time.time()
    for epoch in range(1, CONFIG["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, desc=f"época {epoch} treino")
        val_metrics = evaluate(model, val_loader, loss_fn, device, desc=f"época {epoch} val")
        val_tumor_metrics = evaluate(model, val_tumor_loader, loss_fn, device, desc=f"época {epoch} val(tumor)")
        scheduler.step(val_tumor_metrics["dice"])

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_dice"].append(val_metrics["dice"])
        history["val_iou"].append(val_metrics["iou"])
        history["val_dice_tumor"].append(val_tumor_metrics["dice"])
        history["val_iou_tumor"].append(val_tumor_metrics["iou"])

        print(
            f"[{epoch:02d}/{CONFIG['epochs']}] "
            f"train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_dice={val_metrics['dice']:.4f} val_dice_tumor={val_tumor_metrics['dice']:.4f} "
            f"val_iou_tumor={val_tumor_metrics['iou']:.4f}"
        )

        if val_tumor_metrics["dice"] > best_dice_tumor:
            best_dice_tumor = val_tumor_metrics["dice"]
            epochs_no_improve = 0
            torch.save(
                {"model_state": model.state_dict(), "config": CONFIG, "mean": mean, "std": std, "epoch": epoch},
                OUTPUTS / "checkpoints" / "best_model.pt",
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= CONFIG["early_stop_patience"]:
                print(f"Early stopping na época {epoch} (sem melhora de Dice-tumor por {CONFIG['early_stop_patience']} épocas).")
                break

    elapsed = time.time() - t0
    print(f"Treino concluído em {elapsed / 60:.1f} min. Melhor Dice-tumor (val): {best_dice_tumor:.4f}")

    plot_curves(history, OUTPUTS / "figures" / "training_curves.png")
    with open(OUTPUTS / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    with open(OUTPUTS / "config.json", "w") as f:
        json.dump({**CONFIG, "mean": mean, "std": std, "best_val_dice_tumor": best_dice_tumor, "train_seconds": elapsed}, f, indent=2)

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        df.to_csv(OUTPUTS / f"split_{name}.csv", index=False)

    print("Checkpoint salvo em outputs/checkpoints/best_model.pt")


if __name__ == "__main__":
    main()
