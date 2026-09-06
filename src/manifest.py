"""Constrói o manifesto (lista de pares imagem/máscara) do dataset LGG Brain MRI Segmentation."""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

DATASET_ROOT = Path(__file__).resolve().parent.parent / "datasets" / "kaggle_3m"
MANIFEST_CACHE = Path(__file__).resolve().parent.parent / "outputs" / "manifest.csv"


def _mask_has_tumor(mask_path: Path) -> tuple[bool, float]:
    mask = np.array(Image.open(mask_path))
    positive = mask > 127
    ratio = float(positive.mean())
    return bool(positive.any()), ratio


def build_manifest(root: Path = DATASET_ROOT, use_cache: bool = True) -> pd.DataFrame:
    if use_cache and MANIFEST_CACHE.exists():
        return pd.read_csv(MANIFEST_CACHE)

    rows = []
    patient_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    for patient_dir in patient_dirs:
        image_paths = sorted(
            p for p in patient_dir.glob("*.tif") if "_mask" not in p.name
        )
        for image_path in image_paths:
            mask_path = image_path.with_name(
                image_path.stem + "_mask" + image_path.suffix
            )
            if not mask_path.exists():
                continue
            has_tumor, ratio = _mask_has_tumor(mask_path)
            rows.append(
                {
                    "patient_id": patient_dir.name,
                    "image_path": str(image_path.relative_to(root.parent.parent)),
                    "mask_path": str(mask_path.relative_to(root.parent.parent)),
                    "has_tumor": has_tumor,
                    "tumor_pixel_ratio": ratio,
                }
            )

    df = pd.DataFrame(rows)
    MANIFEST_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MANIFEST_CACHE, index=False)
    return df


if __name__ == "__main__":
    df = build_manifest(use_cache=False)
    print(f"Total de fatias: {len(df)}")
    print(f"Pacientes: {df.patient_id.nunique()}")
    print(f"Fatias com tumor: {df.has_tumor.sum()} ({df.has_tumor.mean():.1%})")
