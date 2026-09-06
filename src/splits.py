"""Divisão treino/validação/teste em nível de PACIENTE (evita vazamento de fatias do
mesmo paciente entre conjuntos diferentes)."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def patient_level_split(
    manifest: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    random_state: int = RANDOM_STATE,
):
    patients = (
        manifest.groupby("patient_id")["has_tumor"].any().reset_index()
    )  # paciente tem ao menos 1 fatia com tumor?

    train_patients, temp_patients = train_test_split(
        patients,
        train_size=train_frac,
        random_state=random_state,
        stratify=patients["has_tumor"],
    )
    val_size_of_temp = val_frac / (1 - train_frac)
    val_patients, test_patients = train_test_split(
        temp_patients,
        train_size=val_size_of_temp,
        random_state=random_state,
        stratify=temp_patients["has_tumor"],
    )

    splits = {}
    for name, subset in [
        ("train", train_patients),
        ("val", val_patients),
        ("test", test_patients),
    ]:
        ids = set(subset["patient_id"])
        splits[name] = manifest[manifest["patient_id"].isin(ids)].reset_index(drop=True)
    return splits


if __name__ == "__main__":
    from manifest import build_manifest

    df = build_manifest()
    splits = patient_level_split(df)
    for name, part in splits.items():
        n_patients = part.patient_id.nunique()
        n_slices = len(part)
        pct_tumor = part.has_tumor.mean()
        print(f"{name:5s}: {n_patients:3d} pacientes, {n_slices:4d} fatias, {pct_tumor:.1%} com tumor")
